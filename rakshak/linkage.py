"""Precision-first cross-document identity resolution and clustering.

Uses deterministic blocking, tiered hard/composite evidence scoring, and
anti-transitive cluster consistency guards to link records across scripts
and hospitals without false merges.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from rakshak.contracts import IdentityClues
from rakshak.translit import GENERIC_NAME_TOKENS, consonant_skeleton, name_tokens_match


@dataclass
class _ClusterState:
    unit_ids: list[str]
    genders: set[str] = field(default_factory=set)
    dobs: set[str] = field(default_factory=set)
    aadhaars: set[str] = field(default_factory=set)
    abhas: set[str] = field(default_factory=set)
    ages: list[int] = field(default_factory=list)
    full_name_tokens: list[tuple[str, ...]] = field(default_factory=list)
    strong_ids: list[str] = field(default_factory=list)


def _has_strong_id(c: IdentityClues) -> bool:
    return bool(
        c.aadhaars or c.abhas or c.phones or c.emails or c.pans
        or c.insurance_ids or c.mrns or c.dob
    )


def _is_patient_bearing(c: IdentityClues) -> bool:
    return bool(c.patient_names or _has_strong_id(c))


def _name_pair_score(a: IdentityClues, b: IdentityClues) -> float:
    best = 0.0
    for na in a.patient_names:
        for nb in b.patient_names:
            sc = name_tokens_match(na.latin_tokens, nb.latin_tokens)
            if sc > best:
                best = sc
    return best


def _rel_pair_score(a: IdentityClues, b: IdentityClues) -> float:
    best = 0.0
    for ra in a.relative_names:
        for rb in b.relative_names:
            sc = name_tokens_match(ra.latin_tokens, rb.latin_tokens)
            if sc > best:
                best = sc
    return best


def _both_multi_token_names(a: IdentityClues, b: IdentityClues) -> bool:
    has_multi_a = any(len(n.latin_tokens) >= 2 for n in a.patient_names)
    has_multi_b = any(len(n.latin_tokens) >= 2 for n in b.patient_names)
    return has_multi_a and has_multi_b


def _age_at_date(dob_iso: str, target_iso: str) -> int | None:
    try:
        by, bm, bd = (int(x) for x in dob_iso.split("-"))
        ty, tm, td = (int(x) for x in target_iso.split("-"))
    except ValueError:
        return None
    return ty - by - (1 if (tm, td) < (bm, bd) else 0)


def _ages_compatible(a: IdentityClues, b: IdentityClues, max_diff: int = 1) -> bool:
    """Compare ages, using DOB + clinical date when available to eliminate multi-year drift."""
    if a.dob and b.age is not None:
        b_dates = [d.iso_date for d in b.dates if d.iso_date and d.role != "dob"]
        if b_dates:
            exp = _age_at_date(a.dob, b_dates[0])
            if exp is not None:
                return abs(exp - b.age) <= 1
    if b.dob and a.age is not None:
        a_dates = [d.iso_date for d in a.dates if d.iso_date and d.role != "dob"]
        if a_dates:
            exp = _age_at_date(b.dob, a_dates[0])
            if exp is not None:
                return abs(exp - a.age) <= 1
    if a.age is not None and b.age is not None:
        return abs(a.age - b.age) <= max_diff
    return False


def _score_pair(a: IdentityClues, b: IdentityClues) -> float:
    """Compute link confidence in [0.0, 1.0]. Returns 0.0 on contradiction or weak evidence."""
    # 1. Hard contradiction gates
    if a.gender and b.gender and a.gender != b.gender:
        return 0.0
    if a.dob and b.dob and a.dob != b.dob:
        return 0.0
    if a.aadhaars and b.aadhaars and set(a.aadhaars).isdisjoint(b.aadhaars):
        return 0.0
    if a.abhas and b.abhas and set(a.abhas).isdisjoint(b.abhas):
        return 0.0
    if a.age is not None and b.age is not None and abs(a.age - b.age) > 3:
        return 0.0

    name_sim = _name_pair_score(a, b)
    both_named = bool(a.patient_names and b.patient_names)
    both_multi = _both_multi_token_names(a, b)

    # If both records have multi-token full names and they strongly disagree, block soft/phone links
    name_conflicts = both_multi and name_sim < 0.72

    # 2. Tier 1: Hard unique identifiers
    if a.aadhaars and b.aadhaars and not set(a.aadhaars).isdisjoint(b.aadhaars):
        return 1.0 if not name_conflicts else 0.0
    if a.abhas and b.abhas and not set(a.abhas).isdisjoint(b.abhas):
        return 1.0 if not name_conflicts else 0.0
    if a.emails and b.emails and not set(a.emails).isdisjoint(b.emails):
        return 0.99 if not name_conflicts else 0.0
    if a.mrns and b.mrns and not set(a.mrns).isdisjoint(b.mrns):
        return 0.98 if not name_conflicts else 0.0
    if a.phones and b.phones and not set(a.phones).isdisjoint(b.phones):
        if both_named and name_sim < 0.70:
            return 0.0
        return 0.97
    if a.pans and b.pans and not set(a.pans).isdisjoint(b.pans):
        return 0.96 if not name_conflicts else 0.0
    if a.insurance_ids and b.insurance_ids and not set(a.insurance_ids).isdisjoint(b.insurance_ids):
        return 0.96 if not name_conflicts else 0.0

    # 3. Tier 2: Composite multi-clue soft links (require strong name + corroborator)
    if not both_named or name_sim < 0.72:
        return 0.0
    if both_multi and name_sim < 0.84:
        return 0.0

    # Extra caution when every token in the name is generic (e.g. "Devi Kumari")
    all_generic = all(
        all(t in GENERIC_NAME_TOKENS for t in n.latin_tokens)
        for n in a.patient_names
    )

    # Corroborator 1: Exact DOB match
    if a.dob and b.dob and a.dob == b.dob:
        return 0.94

    # Corroborator 2: Relative name match ("Asha, W/o Kamal" <-> "आशा राजपूत / कमल राजपूत")
    rel_sim = _rel_pair_score(a, b)
    if rel_sim >= 0.85:
        if all_generic and not (a.village_norm and b.village_norm and a.village_norm == b.village_norm):
            return 0.0
        return 0.92

    # Corroborator 3: Village + compatible Age (uses DOB+visit_date when available, else +/-1)
    if a.village_norm and b.village_norm and a.village_norm == b.village_norm:
        if _ages_compatible(a, b, max_diff=1) and not all_generic:
            return 0.88

    # Corroborator 4: Last-4 Aadhaar + District + Age within +/-2
    if a.aadhaar_last4 and b.aadhaar_last4 and not set(a.aadhaar_last4).isdisjoint(b.aadhaar_last4):
        if a.district_norm and b.district_norm and a.district_norm == b.district_norm:
            return 0.87

    # Corroborator 5: Exact 6-digit PIN + Gender + Age +/-1 (only when full multi-token name matches)
    if both_multi and a.pin and b.pin and a.pin == b.pin and a.gender and b.gender and a.gender == b.gender:
        if a.age is not None and b.age is not None and abs(a.age - b.age) <= 1 and not all_generic:
            return 0.85

    return 0.0


def _build_blocks(clues: list[IdentityClues]) -> dict[str, list[int]]:
    """Build deterministic inverted candidate buckets."""
    buckets: dict[str, list[int]] = defaultdict(list)
    for idx, c in enumerate(clues):
        if not _is_patient_bearing(c):
            continue
        for aa in c.aadhaars:
            buckets[f"aa:{aa}"].append(idx)
        for ab in c.abhas:
            buckets[f"ab:{ab}"].append(idx)
        for em in c.emails:
            buckets[f"em:{em}"].append(idx)
        for ph in c.phones:
            buckets[f"ph:{ph}"].append(idx)
        for mrn in c.mrns:
            buckets[f"mrn:{mrn}"].append(idx)
        for pan in c.pans:
            buckets[f"pan:{pan}"].append(idx)
        for ins in c.insurance_ids:
            buckets[f"ins:{ins}"].append(idx)

        # Soft blocks by name skeletons
        for nc in c.patient_names:
            skels = [consonant_skeleton(t) for t in nc.latin_tokens if t]
            if not skels:
                continue
            first_sk = skels[0]
            last_sk = skels[-1]
            if c.dob:
                buckets[f"ndob:{first_sk}:{c.dob}"].append(idx)
                if last_sk != first_sk:
                    buckets[f"ndob:{last_sk}:{c.dob}"].append(idx)
            if c.village_norm:
                buckets[f"nvill:{first_sk}:{c.village_norm}"].append(idx)
                if last_sk != first_sk:
                    buckets[f"nvill:{last_sk}:{c.village_norm}"].append(idx)
            if c.pin:
                buckets[f"npin:{first_sk}:{c.pin}"].append(idx)
            if c.district_norm:
                for l4 in c.aadhaar_last4:
                    buckets[f"nl4:{first_sk}:{c.district_norm}:{l4}"].append(idx)
            for rc in c.relative_names:
                rskels = [consonant_skeleton(t) for t in rc.latin_tokens if t]
                if rskels:
                    buckets[f"nrel:{first_sk}:{rskels[0]}"].append(idx)
                    if last_sk != first_sk:
                        buckets[f"nrel:{last_sk}:{rskels[0]}"].append(idx)
    return buckets


def _clusters_compatible(ca: _ClusterState, cb: _ClusterState) -> bool:
    """Anti-transitive merge guard across aggregated cluster states."""
    if ca.genders and cb.genders and len(ca.genders | cb.genders) > 1:
        return False
    if ca.dobs and cb.dobs and len(ca.dobs | cb.dobs) > 1:
        return False
    if ca.aadhaars and cb.aadhaars and ca.aadhaars.isdisjoint(cb.aadhaars):
        return False
    if ca.abhas and cb.abhas and ca.abhas.isdisjoint(cb.abhas):
        return False
    if ca.ages and cb.ages:
        lo = min(min(ca.ages), min(cb.ages))
        hi = max(max(ca.ages), max(cb.ages))
        if hi - lo > 3:
            return False
    if ca.full_name_tokens and cb.full_name_tokens:
        # Every pair of multi-token names across the two clusters must be reasonably compatible
        if not any(
            name_tokens_match(ta, tb) >= 0.72
            for ta in ca.full_name_tokens
            for tb in cb.full_name_tokens
        ):
            return False
    return True


def link_records(clues: list[IdentityClues]) -> dict[str, str | None]:
    """Cluster patient-bearing IdentityClues into deterministic secret-independent cluster keys.

    Returns mapping {unit_id: cluster_key_or_None}. Non-patient units and truly
    ambiguous unlinked units map to None.
    """
    n = len(clues)
    parent = list(range(n))
    states: dict[int, _ClusterState] = {}
    for i, c in enumerate(clues):
        strong = sorted(
            list(c.aadhaars) + list(c.abhas) + list(c.phones) + list(c.mrns) + list(c.emails)
        )
        full_toks = [nc.latin_tokens for nc in c.patient_names if len(nc.latin_tokens) >= 2]
        states[i] = _ClusterState(
            unit_ids=[c.unit_id],
            genders={c.gender} if c.gender else set(),
            dobs={c.dob} if c.dob else set(),
            aadhaars=set(c.aadhaars),
            abhas=set(c.abhas),
            ages=[c.age] if c.age is not None else [],
            full_name_tokens=full_toks,
            strong_ids=strong,
        )

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    buckets = _build_blocks(clues)
    candidate_pairs: dict[tuple[int, int], float] = {}
    for bkey in sorted(buckets):
        idxs = sorted(set(buckets[bkey]))
        if len(idxs) < 2 or len(idxs) > 80:
            continue
        for pos_i in range(len(idxs)):
            i = idxs[pos_i]
            for pos_j in range(pos_i + 1, len(idxs)):
                j = idxs[pos_j]
                pair = (i, j)
                if pair in candidate_pairs:
                    continue
                sc = _score_pair(clues[i], clues[j])
                if sc > 0.0:
                    candidate_pairs[pair] = sc

    # Process highest-confidence edges first, breaking ties deterministically
    sorted_edges = sorted(
        candidate_pairs.items(),
        key=lambda kv: (-kv[1], clues[kv[0][0]].unit_id, clues[kv[0][1]].unit_id),
    )

    for (i, j), _sc in sorted_edges:
        ri, rj = _find(i), _find(j)
        if ri == rj:
            continue
        sa, sb = states[ri], states[rj]
        if not _clusters_compatible(sa, sb):
            continue
        # Union into smaller root index deterministically
        keep, drop = (ri, rj) if ri < rj else (rj, ri)
        parent[drop] = keep
        sk, sd = states[keep], states[drop]
        sk.unit_ids.extend(sd.unit_ids)
        sk.genders |= sd.genders
        sk.dobs |= sd.dobs
        sk.aadhaars |= sd.aadhaars
        sk.abhas |= sd.abhas
        sk.ages.extend(sd.ages)
        sk.full_name_tokens.extend(sd.full_name_tokens)
        sk.strong_ids.extend(sd.strong_ids)
        del states[drop]

    # Assign deterministic canonical cluster keys
    result: dict[str, str | None] = {}
    for i, c in enumerate(clues):
        if not _is_patient_bearing(c):
            result[c.unit_id] = None
            continue
        root = _find(i)
        st = states[root]
        # Abstain when a singleton record has only weak clues (e.g. name + village without unique link)
        if len(st.unit_ids) == 1 and not _has_strong_id(c):
            result[c.unit_id] = None
            continue
        if st.strong_ids:
            canon = sorted(set(st.strong_ids))[0]
            result[c.unit_id] = f"ID|{canon}"
        else:
            first_unit = sorted(st.unit_ids)[0]
            result[c.unit_id] = f"UNIT|{first_unit}"
    return result
