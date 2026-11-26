#!/usr/bin/env python3
"""Local adversarial privacy risk simulator modeled on docs/ATTACKS.md. [Person 2]

Evaluates exported sanitized/ + records.jsonl against dev/attacker/registry.csv
and dev/attacker/known_facts.jsonl (A2/A3 record linkage, A4 profile aggregation,
A5 targeted known-fact + global-shift recovery, A6 pseudonym inversion), plus
100% age-band truthfulness and stay-length interval checks against truth.json.

Also reports "QI-only" risk (masking direct PII names/phones/IDs) so Person 2 can
measure Option C's quasi-identifier resistance independently of Person 1's NER.
"""
from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import hashlib
import io
import json
import re
import unittest
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rakshak.clue_extractor import infer_file_date_orders, parse_date_string
from rakshak.contracts import RecordUnit
from rakshak.translit import (
    extract_name_tokens,
    name_tokens_match,
    norm_aadhaar_last4,
    norm_district,
    norm_location,
    norm_phone,
    normalize_digits,
)

_AGE_BAND_RE = re.compile(r"\b(\d{1,3})-(\d{1,3})\b")
_AGE_PLUS_RE = re.compile(r"\b(\d{2,3})\+")
_PIN_FULL_RE = re.compile(r"(?<!\d)([1-9]\d{5})(?!\d)")
_PIN_PREF_RE = re.compile(r"(?<!\d)([1-9]\d{2})\*{3}")
_PSN_RE = re.compile(r"\bPSN-[0-9A-Z]{6,10}\b")


@dataclass
class RegistryPerson:
    rid: str
    name_tokens: tuple[str, ...]
    rel_tokens: tuple[str, ...]
    gender: str
    age: int
    house: str
    village_norm: str
    district_norm: str
    pin: str
    phone: str
    email: str
    last4: str


@dataclass
class ExtractedProfile:
    pseudonym: str
    ages: list[tuple[int, int]] = field(default_factory=list)
    birth_years: set[int] = field(default_factory=set)
    genders: set[str] = field(default_factory=set)
    districts: set[str] = field(default_factory=set)
    villages: set[str] = field(default_factory=set)
    pins: set[str] = field(default_factory=set)
    pin_prefixes: set[str] = field(default_factory=set)
    phones: set[str] = field(default_factory=set)
    emails: set[str] = field(default_factory=set)
    last4s: set[str] = field(default_factory=set)
    names: list[tuple[str, ...]] = field(default_factory=list)
    rel_names: list[tuple[str, ...]] = field(default_factory=list)
    dates: set[str] = field(default_factory=set)
    stay_lengths: set[int] = field(default_factory=set)
    conditions: set[str] = field(default_factory=set)
    hospitals: set[str] = field(default_factory=set)


def load_registry(path: Path) -> list[RegistryPerson]:
    people: list[RegistryPerson] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            people.append(
                RegistryPerson(
                    rid=row["rid"],
                    name_tokens=extract_name_tokens(row["name"]),
                    rel_tokens=extract_name_tokens(row["relative_name"]),
                    gender=row["gender"].strip().upper(),
                    age=int(row["age"]),
                    house=row["house_no"].strip().lower(),
                    village_norm=norm_location(row["village"]),
                    district_norm=norm_district(row["district"]),
                    pin=row["pin"].strip(),
                    phone=norm_phone(row.get("phone", "")) or "",
                    email=row.get("email", "").strip().lower(),
                    last4=row.get("aadhaar_last4", "").strip(),
                )
            )
    return people


def load_known_facts(path: Path) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            facts.append(json.loads(line))
    return facts


def _parse_age_bounds(val: str) -> tuple[int, int] | None:
    s = normalize_digits(val).strip()
    m_plus = _AGE_PLUS_RE.search(s)
    if m_plus:
        lo = int(m_plus.group(1))
        return (lo, 120) if 50 <= lo <= 110 else None
    m_band = _AGE_BAND_RE.search(s)
    if m_band:
        lo, hi = int(m_band.group(1)), int(m_band.group(2))
        if 0 <= lo <= hi <= 120 and (hi - lo) <= 30:
            return (lo, hi)
    m_exact = re.match(r"^(\d{1,3})\b", s)
    if m_exact:
        a = int(m_exact.group(1))
        if 0 <= a <= 120:
            return (a, a)
    return None


def _extract_profile_evidence(
    raw_fields: dict[str, Any],
    raw_text: str,
    file_rel: str,
    date_orders: dict[tuple[str, str], str],
    prof: ExtractedProfile,
    qi_only: bool = False,
) -> None:
    low_rel = file_rel.lower()
    if "sanjeevani" in low_rel:
        prof.hospitals.add("Sanjeevani Multispeciality Hospital")
    elif "jeevanjyoti" in low_rel:
        prof.hospitals.add("Jeevan Jyoti Charitable Hospital")
    elif "arogya" in low_rel:
        prof.hospitals.add("Arogya Seva Sadan")
    elif "dhhardoi" in low_rel or "dhh" in low_rel:
        prof.hospitals.add("District Hospital Hardoi")

    adm_dates: list[dt.date] = []
    dis_dates: list[dt.date] = []

    def _handle_pair(k: str, v: Any) -> None:
        if v is None:
            return
        sv = normalize_digits(str(v)).strip()
        if not sv or sv.startswith("["):
            return
        kl = k.lower()
        if any(x in kl for x in ("age", "umar")):
            b = _parse_age_bounds(sv)
            if b:
                prof.ages.append(b)
        elif any(x in kl for x in ("gender", "sex", "ling")):
            g = sv.upper()[:1]
            if g in ("M", "F"):
                prof.genders.add(g)
        elif "district" in kl or "jila" in kl:
            if "uttar pradesh" not in sv.lower():
                prof.districts.add(norm_district(sv))
        elif "village" in kl or "gaon" in kl or "city" in kl:
            if not qi_only:
                prof.villages.add(norm_location(sv))
        elif "pin" in kl or "zip" in kl or "postal" in kl:
            mp = _PIN_FULL_RE.search(sv)
            mp3 = _PIN_PREF_RE.search(sv)
            if mp:
                prof.pins.add(mp.group(1))
            elif mp3:
                prof.pin_prefixes.add(mp3.group(1))
        elif any(x in kl for x in ("dob", "birth")):
            iso = parse_date_string(sv, date_orders.get((file_rel, k), "dmy"))
            if iso:
                prof.birth_years.add(int(iso[:4]))
        elif any(x in kl for x in ("date", "admission", "discharge", "collected", "registered")):
            iso = parse_date_string(sv, date_orders.get((file_rel, k), "dmy"))
            if iso:
                prof.dates.add(iso)
                if "admis" in kl or kl == "start":
                    adm_dates.append(dt.date.fromisoformat(iso))
                elif "disch" in kl or kl == "end":
                    dis_dates.append(dt.date.fromisoformat(iso))
        elif any(x in kl for x in ("diagnos", "condition", "procedure", "display")) and isinstance(v, str):
            if len(sv) > 3:
                prof.conditions.add(sv.lower())
        elif not qi_only:
            if any(x in kl for x in ("patient_name", "naam")) or kl == "name":
                toks = extract_name_tokens(sv)
                if toks:
                    prof.names.append(toks)
            elif any(x in kl for x in ("relative", "guardian")):
                toks = extract_name_tokens(re.sub(r"(?i)^[sdw]/o\s*", "", sv))
                if toks:
                    prof.rel_names.append(toks)
            elif "phone" in kl or "mobile" in kl:
                ph = norm_phone(sv)
                if ph:
                    prof.phones.add(ph)
            elif "aadhaar" in kl or "aadhar" in kl:
                l4 = norm_aadhaar_last4(sv)
                if l4:
                    prof.last4s.add(l4)

    def _walk(d: dict[str, Any]) -> None:
        for k, v in d.items():
            if isinstance(v, dict):
                _walk(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        _walk(item)
            else:
                _handle_pair(k, v)

    if raw_fields:
        _walk(raw_fields)

    # Free-text extraction
    s = normalize_digits(raw_text)
    for m in re.finditer(r"(?:Age(?:/Sex)?|आयु)\s*[:\-]\s*([^\n,;]+)", s, re.I):
        b = _parse_age_bounds(m.group(1))
        if b:
            prof.ages.append(b)
    for m in re.finditer(r"\b(\d{1,3}(?:-\d{1,3}|\+)?)\s*(?:Yrs?|Y|वर्ष)?\s*/\s*([MF]|पुरुष|महिला)\b", s, re.I):
        b = _parse_age_bounds(m.group(1))
        if b:
            prof.ages.append(b)
        g = "M" if m.group(2).upper().startswith("M") or "पुरुष" in m.group(2) else "F"
        prof.genders.add(g)
    for m in _PIN_PREF_RE.finditer(s):
        prof.pin_prefixes.add(m.group(1))
    if not qi_only:
        for m in _PIN_FULL_RE.finditer(s):
            prof.pins.add(m.group(1))
        for m in re.finditer(r"(?i)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}", s):
            ph = norm_phone(m.group(0))
            if ph:
                prof.phones.add(ph)
    for m in re.finditer(r"(?i)(?:Date of Admission|भर्ती की तारीख)\s*:\s*(\S+)", s):
        iso = parse_date_string(m.group(1), date_orders.get((file_rel, "__doc__"), "dmy"))
        if iso:
            prof.dates.add(iso)
            adm_dates.append(dt.date.fromisoformat(iso))
    for m in re.finditer(r"(?i)(?:Date of Discharge|छुट्टी की तारीख)\s*:\s*(\S+)", s):
        iso = parse_date_string(m.group(1), date_orders.get((file_rel, "__doc__"), "dmy"))
        if iso:
            prof.dates.add(iso)
            dis_dates.append(dt.date.fromisoformat(iso))
    for m in re.finditer(r"(?i)Diagnosis\s*:\s*([^\n]+)", s):
        for part in m.group(1).split(";"):
            clean_c = re.sub(r"\([A-Z]\d+(?:\.\d+)?\)", "", part).strip().lower()
            if len(clean_c) > 3:
                prof.conditions.add(clean_c)

    if adm_dates and dis_dates:
        for a in adm_dates:
            for d in dis_dates:
                if 0 <= (d - a).days <= 90:
                    prof.stay_lengths.add((d - a).days)


def _sweeney_attack(prof: ExtractedProfile, registry: list[RegistryPerson]) -> dict[str, float]:
    """Filter registry sequentially (skipping empty-result filters) and score candidates."""
    cands = list(registry)

    def _try_filter(pred) -> None:
        nonlocal cands
        sub = [p for p in cands if pred(p)]
        if sub:
            cands = sub

    if prof.phones:
        _try_filter(lambda p: p.phone in prof.phones)
    if prof.last4s:
        _try_filter(lambda p: p.last4 in prof.last4s)
    if prof.districts:
        _try_filter(lambda p: p.district_norm in prof.districts)
    if prof.genders:
        _try_filter(lambda p: p.gender in prof.genders)
    if prof.ages:
        lo_all = min(a for a, _ in prof.ages) - 2
        hi_all = max(b for _, b in prof.ages) + 2
        _try_filter(lambda p: lo_all <= p.age <= hi_all)
    if prof.birth_years:
        # Registry age approx relative to 2025/2026
        _try_filter(lambda p: any(abs((2025 - by) - p.age) <= 2 for by in prof.birth_years))
    if prof.villages:
        _try_filter(lambda p: p.village_norm in prof.villages)
    if prof.pins:
        _try_filter(lambda p: p.pin in prof.pins)
    if prof.pin_prefixes:
        _try_filter(lambda p: any(p.pin.startswith(px) for px in prof.pin_prefixes))

    weights: dict[str, float] = {}
    for p in cands:
        w = 1.0
        for nt in prof.names:
            sc = name_tokens_match(nt, p.name_tokens)
            if sc >= 0.9:
                w *= 60.0
                break
            if sc >= 0.75:
                w *= 8.0
                break
        for rt in prof.rel_names:
            if name_tokens_match(rt, p.rel_tokens) >= 0.85:
                w *= 8.0
                break
        weights[p.rid] = w

    tot = sum(weights.values()) or 1.0
    return {rid: w / tot for rid, w in weights.items()}


def run_attack_audit(
    out_dir: Path,
    attacker_dir: Path,
    truth_path: Path,
    qi_only: bool = False,
) -> dict[str, Any]:
    """Run local A2-A6 checks and clinical age-band/stay-length checks."""
    registry = load_registry(attacker_dir / "registry.csv")
    facts = load_known_facts(attacker_dir / "known_facts.jsonl")
    truth = json.loads(truth_path.read_text(encoding="utf-8"))

    # Map truth PTxxxx -> rid and stay_days
    pt_to_rid = {pt: info["rid"] for pt, info in truth["patients"].items()}

    # Load records.jsonl
    rec_map: dict[tuple[str, str], str | None] = {}
    for line in (out_dir / "records.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rel = str(r["file"]).replace("sanitized/", "", 1)
            rec_map[(rel, r["loc"])] = r.get("patient")

    san = out_dir / "sanitized"
    units: list[RecordUnit] = []
    parsed_cache: dict[tuple[str, str], tuple[dict[str, Any], str]] = {}

    for p in sorted(san.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(san).as_posix()
        txt = p.read_text(encoding="utf-8", errors="replace")
        ext = p.suffix.lower()
        if ext in (".csv", ".tsv"):
            delim = "\t" if ext == ".tsv" else ","
            rows = list(csv.reader(io.StringIO(txt, newline=""), delimiter=delim))
            if not rows:
                continue
            header, body = rows[0], rows[1:]
            for i, row in enumerate(body):
                loc = f"row:{i}"
                fd = {header[j]: row[j] for j in range(min(len(header), len(row)))}
                u = RecordUnit(f"{rel}::{loc}", rel, loc, ext[1:], raw_text=delim.join(row), structured_fields=fd)
                units.append(u)
                parsed_cache[(rel, loc)] = (fd, delim.join(row))
        elif ext == ".json":
            data = json.loads(txt)
            is_doc = (
                not isinstance(data, dict)
                or data.get("resourceType") == "Bundle"
                or isinstance(data.get("patient"), dict)
                or "panels" in data
                or "tickets" in data
                or not any(isinstance(v, list) and v and all(isinstance(x, dict) for x in v) for v in data.values())
            )
            if is_doc:
                u = RecordUnit(f"{rel}::doc", rel, "doc", "json", raw_text=txt, structured_fields=data if isinstance(data, dict) else {})
                units.append(u)
                parsed_cache[(rel, "doc")] = (data if isinstance(data, dict) else {}, txt)
            else:
                for k, v in data.items():
                    if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                        pk = k.replace("~", "~0").replace("/", "~1")
                        for i, item in enumerate(v):
                            loc = f"/{pk}/{i}"
                            u = RecordUnit(f"{rel}::{loc}", rel, loc, "json", raw_text=json.dumps(item), structured_fields=item)
                            units.append(u)
                            parsed_cache[(rel, loc)] = (item, json.dumps(item))
        else:
            u = RecordUnit(f"{rel}::doc", rel, "doc", "txt", raw_text=txt)
            units.append(u)
            parsed_cache[(rel, "doc")] = ({}, txt)

    date_orders = infer_file_date_orders(units)

    # Pool evidence per pseudonym (A4) and per patient
    profiles: dict[str, ExtractedProfile] = {}
    pt_to_psn: dict[str, set[str]] = defaultdict(set)
    for rel, f in truth["files"].items():
        for loc, tu in f["units"].items():
            pt = tu.get("patient")
            if not pt or (rel, loc) not in parsed_cache:
                continue
            psn = rec_map.get((rel, loc)) or f"UNLINKED::{rel}::{loc}"
            pt_to_psn[pt].add(psn)
            prof = profiles.setdefault(psn, ExtractedProfile(pseudonym=psn))
            fd, rtxt = parsed_cache[(rel, loc)]
            _extract_profile_evidence(fd, rtxt, rel, date_orders, prof, qi_only=qi_only)

    # A4 Profile re-identification (>= 0.5 confidence) & A2/A3 mean true-person probability
    reid_probs: list[float] = []
    confident_reid = 0
    high_conf_offsets: list[int] = []
    psn_to_rid: dict[str, str] = {}
    for pt, psns in pt_to_psn.items():
        rid = pt_to_rid[pt]
        best_p = 0.0
        for psn in psns:
            psn_to_rid[psn] = rid
            dist = _sweeney_attack(profiles[psn], registry)
            best_p = max(best_p, dist.get(rid, 0.0))
        reid_probs.append(best_p)
        if best_p >= 0.5:
            confident_reid += 1

    # A5 global-shift recovery check (>= 2 profiles with conf >= 0.9 agreeing on offset)
    facts_by_rid = {f["rid"]: f for f in facts}
    for psn, prof in profiles.items():
        dist = _sweeney_attack(prof, registry)
        if not dist:
            continue
        top_rid, top_prob = max(dist.items(), key=lambda x: x[1])
        if top_prob >= 0.9 and top_rid in facts_by_rid and prof.dates:
            for fd_str in facts_by_rid[top_rid].get("visit_dates") or []:
                true_d = dt.date.fromisoformat(fd_str)
                for pd_str in prof.dates:
                    high_conf_offsets.append((dt.date.fromisoformat(pd_str) - true_d).days)

    offset_counts = Counter(high_conf_offsets)
    global_shift_recovered = any(cnt >= 2 for cnt in offset_counts.values())

    # A5 targeted known-facts success (match on rare condition + stay length + hospital + shift-invariant checks)
    cond_counts = Counter(c for p in profiles.values() for c in p.conditions)
    targeted_hits = 0
    total_real_targets = 0
    for f in facts:
        rid = f["rid"]
        if rid not in set(pt_to_rid.values()):
            continue
        total_real_targets += 1
        f_cond = (f.get("condition_hint") or "").lower()
        f_stay = f.get("stay_days")
        f_hosp = f.get("hospital", "")
        scores: dict[str, float] = {}
        for psn, prof in profiles.items():
            # Consistent with registry person?
            dist = _sweeney_attack(prof, registry)
            base_prob = dist.get(rid, 0.0)
            if base_prob == 0.0:
                continue
            sig = 0.0
            if f_cond and any(f_cond in c or c in f_cond for c in prof.conditions):
                is_rare = any(cond_counts[c] <= 3 for c in prof.conditions if f_cond in c or c in f_cond)
                sig += 8.0 if is_rare else 1.5
            if f_stay is not None and f_stay in prof.stay_lengths:
                sig += 3.0
            if sig > 0:
                if f_hosp and f_hosp in prof.hospitals:
                    sig *= 2.0
                scores[psn] = base_prob * sig
        if scores:
            tot = sum(scores.values())
            true_psns = {psn for psn, r in psn_to_rid.items() if r == rid}
            true_mass = sum(v for k, v in scores.items() if k in true_psns) / tot
            if true_mass >= 0.5:
                targeted_hits += 1

    # A6 Pseudonym inversion check
    inverted = 0
    for pt, info in truth["patients"].items():
        originals = [pt, info["rid"], info["pid"]] + [u for ul in info["uhids"].values() for u in ul]
        hashes = set()
        for orig in originals:
            for variant in (orig, orig.lower(), orig.upper(), re.sub(r"\D", "", orig)):
                if not variant:
                    continue
                b = variant.encode("utf-8")
                hashes.add(hashlib.md5(b).hexdigest()[:8])
                hashes.add(hashlib.sha1(b).hexdigest()[:8])
                hashes.add(hashlib.sha256(b).hexdigest()[:8])
                hashes.add(base64.b64encode(b).decode("ascii")[:8])
        for psn in pt_to_psn.get(pt, set()):
            if psn.startswith("PSN-"):
                code = psn[4:].lower()
                if any(h.lower() in code or code in h.lower() for h in hashes):
                    inverted += 1

    # Clinical Age-Band truthfulness & Stay-Length exactness checks
    bad_age_bands = 0
    checked_ages = 0
    for rel, f in truth["files"].items():
        for loc, tu in f["units"].items():
            if (rel, loc) not in parsed_cache:
                continue
            fd, rtxt = parsed_cache[(rel, loc)]
            for fld in tu.get("fields") or []:
                if fld["kind"] == "age":
                    true_age = fld["true"]
                    # Find rendered age bound in output
                    for k, v in fd.items():
                        if "age" in k.lower() or "umar" in k.lower():
                            b = _parse_age_bounds(str(v))
                            if b:
                                checked_ages += 1
                                if not (b[0] <= true_age <= b[1]):
                                    bad_age_bands += 1

    stay_checked = 0
    stay_bad = 0
    for rel, f in truth["files"].items():
        for loc, tu in f["units"].items():
            if (rel, loc) not in parsed_cache:
                continue
            adm_t = [dt.date.fromisoformat(d["true"]) for d in tu.get("dates") or [] if d["event"] == "admission"]
            dis_t = [dt.date.fromisoformat(d["true"]) for d in tu.get("dates") or [] if d["event"] == "discharge"]
            if adm_t and dis_t:
                true_stay = (dis_t[0] - adm_t[0]).days
                psn = rec_map.get((rel, loc))
                if psn and psn in profiles:
                    stay_checked += 1
                    if true_stay not in profiles[psn].stay_lengths:
                        stay_bad += 1

    n_pts = max(1, len(reid_probs))
    return {
        "mode": "qi_only" if qi_only else "full_output",
        "mean_reid_prob": round(sum(reid_probs) / n_pts, 4),
        "a4_patients_reidentified": round(confident_reid / n_pts, 4),
        "a5_global_shift_recovered": global_shift_recovered,
        "a5_targeted_success": round(targeted_hits / max(1, total_real_targets), 4),
        "a6_pseudonym_inverted": inverted,
        "age_bands_checked": checked_ages,
        "age_bands_invalid": bad_age_bands,
        "stays_checked": stay_checked,
        "stays_mismatched": stay_bad,
    }


class TestPhase6AttackRisk(unittest.TestCase):
    def test_sweeney_filter_and_age_band_parser(self) -> None:
        self.assertEqual(_parse_age_bounds("45-49"), (45, 49))
        self.assertEqual(_parse_age_bounds("80+"), (80, 120))
        self.assertEqual(_parse_age_bounds("85+"), (85, 120))
        reg = [
            RegistryPerson("R1", ("asha", "rajput"), ("kamal",), "F", 48, "12", "sltn", "sltn", "228355", "", "", ""),
            RegistryPerson("R2", ("usha", "tiwari"), ("rahul",), "F", 47, "99", "sltn", "sltn", "228355", "", "", ""),
            RegistryPerson("R3", ("neha", "paswan"), ("deepak",), "F", 46, "55", "sltn", "sltn", "228355", "", "", ""),
        ]
        prof = ExtractedProfile("PSN-TEST01", ages=[(45, 49)], genders={"F"}, districts={"sltn"})
        dist = _sweeney_attack(prof, reg)
        # All 3 match (45-49, F, sltn) -> max probability is 1/3 (< 0.5 threshold)
        self.assertLess(dist["R1"], 0.5)


def main() -> int:
    ap = argparse.ArgumentParser(description="Local adversarial privacy risk simulator")
    ap.add_argument("--sanitized", required=True, help="Output folder containing sanitized/ and records.jsonl")
    ap.add_argument("--attacker", required=True, help="Path to dev/attacker")
    ap.add_argument("--truth", required=True, help="Path to dev/truth/truth.json")
    args = ap.parse_args()

    out_dir = Path(args.sanitized)
    att_dir = Path(args.attacker)
    truth_path = Path(args.truth)

    full = run_attack_audit(out_dir, att_dir, truth_path, qi_only=False)
    qi = run_attack_audit(out_dir, att_dir, truth_path, qi_only=True)
    print(json.dumps({"full_output": full, "qi_only_option_c": qi}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
