"""Quasi-identifier privacy layer (Option C: Adaptive k-Anonymity Widening).

Provides:
  * Keyed HMAC-SHA256 pseudonyms (PSN-XXXXXX) and non-zero per-patient date shifts
  * Truthful age banding (5-yr standard; 10/20-yr widened for small/rare cells)
  * Equivalence-class + rare-condition analysis for adaptive widening/QI suppression
  * Format-preserving date shifting and quasi-identifier transforms for CSV/JSON/FHIR/TXT
"""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import re
from collections import Counter, defaultdict
from typing import Any

from rakshak.clue_extractor import _classify_key, parse_date_string
from rakshak.contracts import IdentityClues, PolicyConfig, PrivacyPlan
from rakshak.translit import normalize_digits

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford base32: no I, L, O, U

_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})(T[^\s\"']*)?\b")
_SLASH_DATE_RE = re.compile(r"\b(\d{1,2})([/.-])(\d{1,2})\2(\d{2}|\d{4})\b")
_TEXT_MONTH_DATE_RE = re.compile(
    r"\b(\d{1,2})\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"(?:\s+(\d{4}))?\b",
    re.I,
)
_MONTH_MAP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# Free-text age forms matching attacker's parser (never match lab ranges, history durations, or already-banded "35-39" / "85+")
_TXT_AGE_YRS_SEX_RE = re.compile(
    r"(?<!-)\b(\d{1,3})\b(?!\s*[-+])(\s*(?:Yrs?|Y|वर्ष)\s*/\s*(?:[MF]|पुरुष|महिला))\b",
    re.I,
)
_TXT_AGE_SLASH_SEX_RE = re.compile(r"(?<!-)\b(\d{1,3})\b(?!\s*[-+])(\s*/\s*[MF])\b")
_TXT_AGE_ANCHOR_RE = re.compile(
    r"((?:Age(?:/Sex)?|आयु)\s*[:\-]\s*)(?<!-)\b(\d{1,3})\b(?!\s*[-+])(\s*(?:Yrs?|Y|वर्ष)?)",
    re.I,
)
_TXT_DATE_LINE_RE = re.compile(
    r"(?im)^(.*?(?:Date\s+of\s+(?:Admission|Discharge)|Date|Visit|भर्ती\s+की\s+तारीख|छुट्टी\s+की\s+तारीख|तारीख|DOB|जन्म).*)$"
)
_TXT_WA_HEADER_RE = re.compile(r"^(\[)(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})(,[^\]]*\])", re.M)


def make_pseudonym(cluster_key: str, secret: bytes) -> str:
    """Stable, non-reversible HMAC-SHA256 pseudonym keyed by RAKSHA_SECRET."""
    digest = hmac.new(secret, cluster_key.encode("utf-8"), hashlib.sha256).digest()
    n = int.from_bytes(digest[:5], "big")
    code = "".join(_ALPHABET[(n >> (5 * i)) & 31] for i in range(6))
    return f"PSN-{code}"


def make_date_shift_days(key: str, secret: bytes, max_days: int = 180) -> int:
    """Deterministic non-zero per-patient date shift in [-max_days, +max_days]."""
    m = max(1, int(max_days))
    digest = hmac.new(secret, b"shift|" + key.encode("utf-8"), hashlib.sha256).digest()
    val = int.from_bytes(digest[:4], "big") % (2 * m) - m
    return val if val != 0 else 1


def format_age_band(age: int, width: int = 5, cap: int = 85) -> str:
    """Return a truthful age band guaranteed to contain `age`."""
    if age < 0:
        return "0-4"
    if age >= cap:
        return f"{cap}+"
    lo = (age // width) * width
    hi = min(lo + width - 1, cap - 1) if lo < cap else lo + width - 1
    return f"{lo}-{hi}"


def shift_date_token(raw_date: str, shift_days: int, order: str = "dmy") -> str:
    """Shift a single date string while preserving its surface format."""
    s = normalize_digits(raw_date).strip()
    m_iso = _ISO_DATE_RE.search(s)
    if m_iso:
        y, mo, d = int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))
        try:
            shifted = dt.date(y, mo, d) + dt.timedelta(days=shift_days)
            suffix = m_iso.group(4) or ""
            return s[:m_iso.start()] + shifted.isoformat() + suffix + s[m_iso.end():]
        except ValueError:
            return raw_date

    m_sl = _SLASH_DATE_RE.search(s)
    if m_sl:
        p1, sep, p2, y_raw = int(m_sl.group(1)), m_sl.group(2), int(m_sl.group(3)), m_sl.group(4)
        y = 2000 + int(y_raw) if len(y_raw) == 2 else int(y_raw)
        iso = parse_date_string(m_sl.group(0), order)
        if iso:
            shifted = dt.date.fromisoformat(iso) + dt.timedelta(days=shift_days)
            yy = f"{shifted.year % 100:02d}" if len(y_raw) == 2 else f"{shifted.year:04d}"
            if order == "mdy":
                rep = f"{shifted.month:02d}{sep}{shifted.day:02d}{sep}{yy}"
            else:
                rep = f"{shifted.day:02d}{sep}{shifted.month:02d}{sep}{yy}"
            return s[:m_sl.start()] + rep + s[m_sl.end():]

    m_txt = _TEXT_MONTH_DATE_RE.search(s)
    if m_txt:
        d = int(m_txt.group(1))
        mo = _MONTH_MAP[m_txt.group(2).lower()]
        y = int(m_txt.group(3)) if m_txt.group(3) else 2026
        try:
            shifted = dt.date(y, mo, d) + dt.timedelta(days=shift_days)
            if m_txt.group(3):
                rep = f"{shifted.day} {_MONTH_NAMES[shifted.month]} {shifted.year}"
            else:
                rep = f"{shifted.day} {_MONTH_NAMES[shifted.month]}"
            return s[:m_txt.start()] + rep + s[m_txt.end():]
        except ValueError:
            return raw_date

    return raw_date


def build_privacy_plans(
    clues: list[IdentityClues],
    cluster_map: dict[str, str | None],
    policy: PolicyConfig,
    secret: bytes,
) -> dict[str, PrivacyPlan]:
    """Compute Option C adaptive k-anonymity + rare-condition PrivacyPlans for all units."""
    # Group units by patient cluster
    cluster_units: dict[str, list[IdentityClues]] = defaultdict(list)
    for c in clues:
        ck = cluster_map.get(c.unit_id)
        if ck:
            cluster_units[ck].append(c)

    # Build patient-level profiles and condition frequencies
    profile_age: dict[str, int | None] = {}
    profile_gender: dict[str, str] = {}
    profile_district: dict[str, str] = {}
    profile_conds: dict[str, set[str]] = {}
    cond_patient_counts: Counter[str] = Counter()

    for ck, members in sorted(cluster_units.items()):
        ages = [m.age for m in members if m.age is not None]
        genders = [m.gender for m in members if m.gender]
        dists = [m.district_norm for m in members if m.district_norm]
        conds = {cond for m in members for cond in m.conditions if cond}

        profile_age[ck] = max(ages) if ages else None
        profile_gender[ck] = genders[0] if genders else "U"
        profile_district[ck] = dists[0] if dists else "U"
        profile_conds[ck] = conds
        for cond in conds:
            cond_patient_counts[cond] += 1

    # Equivalence class counts over standard (5-yr age band, gender, district)
    std_cells: Counter[tuple[str, str, str]] = Counter()
    profile_std_band: dict[str, str | None] = {}
    for ck in sorted(cluster_units):
        age = profile_age[ck]
        band = format_age_band(age, policy.age_band_years, policy.max_age_cap) if age is not None else "UNK"
        profile_std_band[ck] = band if age is not None else None
        std_cells[(band, profile_gender[ck], profile_district[ck])] += 1

    # Widened 10-year cell counts
    wide10_cells: Counter[tuple[str, str, str]] = Counter()
    for ck in sorted(cluster_units):
        age = profile_age[ck]
        band10 = format_age_band(age, 10, 80) if age is not None else "UNK"
        wide10_cells[(band10, profile_gender[ck], profile_district[ck])] += 1

    # Decide per-cluster generalization tier under Option C
    cluster_tier: dict[str, tuple[str | None, bool, bool, str]] = {}
    for ck in sorted(cluster_units):
        age = profile_age[ck]
        std_band = profile_std_band[ck]
        has_rare = any(
            cond_patient_counts[cond] <= policy.rare_condition_threshold
            for cond in profile_conds[ck]
        )
        cell_k = std_cells[(std_band or "UNK", profile_gender[ck], profile_district[ck])]
        is_elderly = age is not None and age >= policy.max_age_cap

        if not policy.adaptive_widening:
            cluster_tier[ck] = (std_band, True, True, "standard")
            continue

        if has_rare or (cell_k < policy.min_group_size and is_elderly):
            # High-risk outlier: 20-year age band (or 80+), suppress PIN prefix and District
            wide_band = format_age_band(age, 20, 80) if age is not None else None
            cluster_tier[ck] = (wide_band, False, False, "suppressed")
        elif cell_k < policy.min_group_size or is_elderly:
            # Small cell: widen to 10-year band, drop PIN prefix; if still k<min_group_size, suppress District too
            band10 = format_age_band(age, 10, 80) if age is not None else None
            k10 = wide10_cells[(band10 or "UNK", profile_gender[ck], profile_district[ck])]
            if k10 < policy.min_group_size:
                wide20 = format_age_band(age, 20, 80) if age is not None else None
                cluster_tier[ck] = (wide20, False, False, "suppressed")
            else:
                cluster_tier[ck] = (band10, True, False, "widened")
        else:
            cluster_tier[ck] = (std_band, True, True, "standard")

    # Assemble per-unit PrivacyPlan
    plans: dict[str, PrivacyPlan] = {}
    for c in clues:
        ck = cluster_map.get(c.unit_id)
        if ck:
            pid = make_pseudonym(ck, secret)
            shift = make_date_shift_days(ck, secret, policy.date_shift_max_days)
            p_band, keep_dist, keep_pin, level = cluster_tier[ck]
            # Use unit-specific age band when unit has its own age (e.g. 2024 vs 2026 visit)
            if c.age is not None:
                if level == "standard":
                    unit_band: str | None = format_age_band(c.age, policy.age_band_years, policy.max_age_cap)
                elif level == "widened":
                    unit_band = format_age_band(c.age, 10, 80)
                else:
                    unit_band = format_age_band(c.age, 20, 80)
            else:
                unit_band = p_band
            plans[c.unit_id] = PrivacyPlan(
                unit_id=c.unit_id,
                cluster_key=ck,
                pseudonym=pid,
                date_shift_days=shift,
                age_band=unit_band,
                keep_district=keep_dist,
                keep_pin_prefix=keep_pin,
                generalization_level=level,
            )
        else:
            # Unlinked or non-patient unit: deterministic per-unit date shift, standard 5-yr age band if age present
            shift = make_date_shift_days(c.unit_id, secret, policy.date_shift_max_days)
            unit_band = (
                format_age_band(c.age, policy.age_band_years, policy.max_age_cap)
                if c.age is not None
                else None
            )
            plans[c.unit_id] = PrivacyPlan(
                unit_id=c.unit_id,
                cluster_key=None,
                pseudonym=None,
                date_shift_days=shift,
                age_band=unit_band,
                keep_district=True,
                keep_pin_prefix=True,
                generalization_level="standard",
            )
    return plans


def generalize_pin(raw_pin: str, keep_prefix: bool, prefix_digits: int = 3) -> str:
    """Keep 3-digit PIN prefix (e.g. 226***) when allowed, else suppress."""
    digits = re.sub(r"\D", "", normalize_digits(raw_pin))
    if not digits:
        return "[PIN]"
    if keep_prefix and len(digits) >= prefix_digits:
        return digits[:prefix_digits] + "*" * (max(6, len(digits)) - prefix_digits)
    return "[PIN]"


def apply_privacy_to_text(
    text: str,
    plan: PrivacyPlan,
    date_order: str = "dmy",
) -> tuple[str, bool]:
    """Shift clinical/DOB dates and band ages in free text. Returns (new_text, changed)."""
    out = text
    changed = False

    # 1. Shift WhatsApp bracketed date headers [DD/MM/YY, ...]
    def _wa_sub(m: re.Match) -> str:
        nonlocal changed
        changed = True
        return f"{m.group(1)}{shift_date_token(m.group(2), plan.date_shift_days, date_order)}{m.group(3)}"

    out = _TXT_WA_HEADER_RE.sub(_wa_sub, out)

    # 2. Shift dated lines (Admission, Discharge, Visit, DOB, तारीख, etc.)
    def _line_sub(m: re.Match) -> str:
        nonlocal changed
        line = m.group(1)
        if re.search(r"(?i)\bDOB\b|जन्म", line) and plan.generalization_level != "standard":
            changed = True
            return re.sub(r"\b\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b", "[DOB]", line)
        new_line = shift_date_token(line, plan.date_shift_days, date_order)
        if new_line != line:
            changed = True
        return new_line

    out = _TXT_DATE_LINE_RE.sub(_line_sub, out)

    # 3. Replace exact age expressions with truthful age band
    if plan.age_band:
        def _age_yrs_sub(m: re.Match) -> str:
            nonlocal changed
            age_val = int(normalize_digits(m.group(1)))
            if 0 <= age_val <= 120:
                changed = True
                return f"{plan.age_band}{m.group(2)}"
            return m.group(0)

        def _age_slash_sub(m: re.Match) -> str:
            nonlocal changed
            age_val = int(normalize_digits(m.group(1)))
            if 0 <= age_val <= 120:
                changed = True
                return f"{plan.age_band}{m.group(2)}"
            return m.group(0)

        def _age_anchor_sub(m: re.Match) -> str:
            nonlocal changed
            age_val = int(normalize_digits(m.group(2)))
            if 0 <= age_val <= 120:
                changed = True
                return f"{m.group(1)}{plan.age_band}{m.group(3)}"
            return m.group(0)

        out = _TXT_AGE_YRS_SEX_RE.sub(_age_yrs_sub, out)
        out = _TXT_AGE_SLASH_SEX_RE.sub(_age_slash_sub, out)
        out = _TXT_AGE_ANCHOR_RE.sub(_age_anchor_sub, out)

    if not plan.keep_district:
        out_d = re.sub(r"(?i)((?:Distt?\.?|जिला)\s*[:.]?\s*)([^\n,।-]+)", r"\1Uttar Pradesh", out)
        if out_d != out:
            out = out_d
            changed = True

    return out, changed


def apply_privacy_to_structured_record(
    fields: dict[str, Any],
    plan: PrivacyPlan,
    file_rel: str,
    date_orders: dict[tuple[str, str], str],
    policy: PolicyConfig,
) -> tuple[dict[str, Any], bool]:
    """Transform quasi-identifiers (age, dob, dates, village, district, pin) in a dict."""
    changed = False

    def _transform(d: dict[str, Any]) -> dict[str, Any]:
        nonlocal changed
        res: dict[str, Any] = {}
        for k, v in d.items():
            if k == "_export_meta":
                res[k] = v
                continue
            role = _classify_key(k)
            if isinstance(v, dict):
                res[k] = _transform(v)
                continue
            if isinstance(v, list):
                res[k] = [_transform(x) if isinstance(x, dict) else x for x in v]
                continue
            if v is None or v == "":
                res[k] = v
                continue

            sv = str(v)
            if role == "age" and plan.age_band:
                # Never overwrite negative synthetic placeholder ages (e.g. "-2 वर्ष") with "0-19"
                if sv.lstrip().startswith("-"):
                    res[k] = sv
                else:
                    res[k] = plan.age_band
                    changed = True
            elif role == "age_sex" and plan.age_band:
                m = _TXT_AGE_SLASH_SEX_RE.search(sv)
                if m:
                    res[k] = f"{plan.age_band}{m.group(2)}"
                else:
                    res[k] = plan.age_band
                changed = True
            elif role == "dob":
                if plan.generalization_level != "standard":
                    res[k] = "[DOB]"
                else:
                    order = date_orders.get((file_rel, k), "dmy")
                    res[k] = shift_date_token(sv, plan.date_shift_days, order)
                changed = True
            elif role == "date":
                order = date_orders.get((file_rel, k), "dmy")
                res[k] = shift_date_token(sv, plan.date_shift_days, order)
                changed = True
            elif role == "village":
                res[k] = "[VILLAGE]"
                changed = True
            elif role == "district":
                res[k] = sv if plan.keep_district else "Uttar Pradesh"
                if not plan.keep_district:
                    changed = True
            elif role == "pin":
                res[k] = generalize_pin(sv, plan.keep_pin_prefix, policy.pin_prefix_digits)
                changed = True
            elif role == "free_text":
                new_t, txt_ch = apply_privacy_to_text(sv, plan, date_orders.get((file_rel, "__doc__"), "dmy"))
                res[k] = new_t
                changed = changed or txt_ch
            else:
                res[k] = v
        return res

    return _transform(fields), changed


def apply_privacy_to_fhir_bundle(
    bundle: dict[str, Any],
    plan: PrivacyPlan,
    policy: PolicyConfig,
) -> tuple[dict[str, Any], bool]:
    """Shift dates and generalize Patient demographics inside a FHIR R4 Bundle."""
    changed = False
    for entry in bundle.get("entry") or []:
        if not isinstance(entry, dict):
            continue
        res = entry.get("resource")
        if not isinstance(res, dict):
            continue
        rtype = res.get("resourceType")
        if rtype == "Patient":
            if res.get("birthDate"):
                if plan.generalization_level != "standard":
                    # Year-only or shifted year on high-risk profiles (keep valid FHIR date syntax: YYYY)
                    iso = parse_date_string(str(res["birthDate"]), "iso")
                    res["birthDate"] = iso[:4] if iso else "1900"
                else:
                    res["birthDate"] = shift_date_token(str(res["birthDate"]), plan.date_shift_days, "iso")
                changed = True
            for addr in res.get("address") or []:
                if isinstance(addr, dict):
                    if "line" in addr:
                        addr["line"] = ["[ADDRESS]"]
                        changed = True
                    if "city" in addr:
                        addr["city"] = "[VILLAGE]"
                        changed = True
                    if "district" in addr and not plan.keep_district:
                        addr["district"] = "Uttar Pradesh"
                        changed = True
                    if "postalCode" in addr:
                        addr["postalCode"] = generalize_pin(
                            str(addr["postalCode"]), plan.keep_pin_prefix, policy.pin_prefix_digits
                        )
                        changed = True
        elif rtype == "Encounter":
            per = res.get("period")
            if isinstance(per, dict):
                for edge in ("start", "end"):
                    if per.get(edge):
                        per[edge] = shift_date_token(str(per[edge]), plan.date_shift_days, "iso")
                        changed = True
        elif rtype == "Composition":
            if res.get("date"):
                res["date"] = shift_date_token(str(res["date"]), plan.date_shift_days, "iso")
                changed = True
    # Also shift Bundle-level timestamp / meta.lastUpdated so dates remain internally consistent
    if bundle.get("timestamp"):
        bundle["timestamp"] = shift_date_token(str(bundle["timestamp"]), plan.date_shift_days, "iso")
        changed = True
    if isinstance(bundle.get("meta"), dict) and bundle["meta"].get("lastUpdated"):
        bundle["meta"]["lastUpdated"] = shift_date_token(
            str(bundle["meta"]["lastUpdated"]), plan.date_shift_days, "iso"
        )
        changed = True
    return bundle, changed
