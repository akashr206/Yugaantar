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
_MONTH_TOKEN_ALT = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|"
    r"जनवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|सितम्बर|अक्टूबर|अक्तूबर|नवंबर|नवम्बर|दिसंबर|दिसम्बर"
)
_TEXT_MONTH_DATE_RE = re.compile(
    rf"(?<!\d)(\d{{1,2}})(?:st|nd|rd|th)?([\s/-]+)({_MONTH_TOKEN_ALT})(?:(?:,?\s+|\2)(\d{{2,4}}))?(?!\d)",
    re.I,
)
_US_TEXT_MONTH_DATE_RE = re.compile(
    rf"\b({_MONTH_TOKEN_ALT})([\s,/-]+)(\d{{1,2}})(?:st|nd|rd|th)?(?:(?:,?\s+|[\s/-]+)(\d{{2,4}}))?(?!\d)(?!\s*(?:mg|mcg|gm?|ml|iu|/ul|mmhg|yrs?|years?|days?|wks?|weeks?|months?|hrs?|%)\b)",
    re.I,
)
_YMD_TEXT_MONTH_DATE_RE = re.compile(
    rf"(?<!\d)(\d{{4}})([\s/-]+)({_MONTH_TOKEN_ALT})\2(\d{{1,2}})(?:st|nd|rd|th)?(?!\d)",
    re.I,
)
_MONTH_MAP = {
    "jan": 1, "january": 1, "जनवरी": 1,
    "feb": 2, "february": 2, "फरवरी": 2,
    "mar": 3, "march": 3, "मार्च": 3,
    "apr": 4, "april": 4, "अप्रैल": 4,
    "may": 5, "मई": 5,
    "jun": 6, "june": 6, "जून": 6,
    "jul": 7, "july": 7, "जुलाई": 7,
    "aug": 8, "august": 8, "अगस्त": 8,
    "sep": 9, "sept": 9, "september": 9, "सितंबर": 9, "सितम्बर": 9,
    "oct": 10, "october": 10, "अक्टूबर": 10, "अक्तूबर": 10,
    "nov": 11, "november": 11, "नवंबर": 11, "नवम्बर": 11,
    "dec": 12, "december": 12, "दिसंबर": 12, "दिसम्बर": 12,
}
_MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
_MONTH_NAMES_SHORT = [
    "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
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
    """Stable, non-reversible 50-bit HMAC-SHA256 pseudonym keyed by RAKSHA_SECRET."""
    digest = hmac.new(secret, cluster_key.encode("utf-8"), hashlib.sha256).digest()
    n = int.from_bytes(digest[:7], "big")
    code = "".join(_ALPHABET[(n >> (5 * i)) & 31] for i in range(10))
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

    def _expand_yr(y_str: str) -> int:
        yi = int(y_str)
        if len(y_str) >= 4:
            return yi
        return 2000 + yi if yi <= 35 else 1900 + yi

    def _fmt_month(mo_num: int, orig_tok: str) -> str:
        if orig_tok.isascii() and len(orig_tok) <= 4:
            return _MONTH_NAMES_SHORT[mo_num]
        return _MONTH_NAMES[mo_num]

    def _fmt_year(y_val: int, orig_yr: str) -> str:
        return f"{y_val % 100:02d}" if len(orig_yr) == 2 else f"{y_val:04d}"

    m_ymdt = _YMD_TEXT_MONTH_DATE_RE.search(s)
    if m_ymdt:
        y = int(m_ymdt.group(1))
        sep = m_ymdt.group(2)
        mo_tok = m_ymdt.group(3)
        mo = _MONTH_MAP.get(mo_tok.lower())
        d = int(m_ymdt.group(4))
        if mo:
            try:
                shifted = dt.date(y, mo, d) + dt.timedelta(days=shift_days)
                rep = f"{shifted.year:04d}{sep}{_fmt_month(shifted.month, mo_tok)}{sep}{shifted.day:02d}"
                return s[:m_ymdt.start()] + rep + s[m_ymdt.end():]
            except ValueError:
                return raw_date

    m_txt = _TEXT_MONTH_DATE_RE.search(s)
    if m_txt:
        d = int(m_txt.group(1))
        sep = m_txt.group(2)
        mo_tok = m_txt.group(3)
        yr_tok = m_txt.group(4)
        if not yr_tok and mo_tok == "may":
            return raw_date
        mo = _MONTH_MAP.get(mo_tok.lower())
        y = _expand_yr(yr_tok) if yr_tok else 2026
        if mo:
            try:
                shifted = dt.date(y, mo, d) + dt.timedelta(days=shift_days)
                out_sep = sep if sep in ("-", "/") else " "
                m_str = _fmt_month(shifted.month, mo_tok)
                if yr_tok:
                    y_str = _fmt_year(shifted.year, yr_tok)
                    rep = f"{shifted.day:02d}{out_sep}{m_str}{out_sep}{y_str}" if out_sep in ("-", "/") else f"{shifted.day} {m_str} {y_str}"
                else:
                    rep = f"{shifted.day}{out_sep}{m_str}"
                return s[:m_txt.start()] + rep + s[m_txt.end():]
            except ValueError:
                return raw_date

    m_us = _US_TEXT_MONTH_DATE_RE.search(s)
    if m_us:
        mo_tok = m_us.group(1)
        sep = m_us.group(2)
        d = int(m_us.group(3))
        yr_tok = m_us.group(4)
        if not yr_tok and mo_tok == "may":
            return raw_date
        mo = _MONTH_MAP.get(mo_tok.lower())
        y = _expand_yr(yr_tok) if yr_tok else 2026
        if mo:
            try:
                shifted = dt.date(y, mo, d) + dt.timedelta(days=shift_days)
                m_str = _fmt_month(shifted.month, mo_tok)
                if yr_tok:
                    y_str = _fmt_year(shifted.year, yr_tok)
                    if sep in ("-", "/"):
                        rep = f"{m_str}{sep}{shifted.day:02d}{sep}{y_str}"
                    else:
                        rep = f"{m_str} {shifted.day}, {y_str}"
                else:
                    out_sep = sep if sep in ("-", "/") else " "
                    rep = f"{m_str}{out_sep}{shifted.day}"
                return s[:m_us.start()] + rep + s[m_us.end():]
            except ValueError:
                return raw_date

    return raw_date


def _candidate_wide_bands(ages: list[int]) -> list[tuple[str, int, int]]:
    """Return all truthful width<=30 or N+ bands covering every observed age in `ages`."""
    lo_a = max(0, min(ages))
    hi_a = min(120, max(ages))
    candidates: list[tuple[str, int, int]] = []
    for base in range(0, 80, 10):
        b_lo, b_hi = base, base + 29
        if b_lo <= lo_a and hi_a <= b_hi:
            candidates.append((f"{b_lo}-{b_hi}", b_lo, b_hi))
    for cap in (50, 60, 70, 80):
        if lo_a >= cap:
            candidates.append((f"{cap}+", cap, 120))
    if not candidates:
        fb = _hull_age_band(ages, 20, 80)
        if fb.endswith("+"):
            c = int(fb[:-1])
            candidates.append((fb, c, 120))
        elif "-" in fb:
            a, b = (int(x) for x in fb.split("-"))
            candidates.append((fb, a, b))
        else:
            candidates.append((f"{(lo_a // 10) * 10}+", (lo_a // 10) * 10, 120))
    return candidates


def _hull_age_band(ages: list[int], width: int, cap: int) -> str:
    """Return a single truthful band covering [min(ages), max(ages)] with width <= 30."""
    lo_a = max(0, min(ages))
    hi_a = min(120, max(ages))
    if lo_a >= cap:
        return f"{cap}+"
    if hi_a >= cap and lo_a >= 50:
        step = (lo_a // 10) * 10
        return f"{max(50, step)}+"
    b_lo = format_age_band(lo_a, width, cap)
    b_hi = format_age_band(hi_a, width, cap)
    if b_lo == b_hi:
        return b_lo
    span = hi_a - lo_a
    for w in (10, 20, 30):
        lo_w = (lo_a // w) * w
        hi_w = lo_w + w - 1
        if lo_w <= lo_a and hi_a <= hi_w and (hi_w - lo_w) <= 30:
            return f"{lo_w}-{min(120, hi_w)}"
    lo_c = (lo_a // 5) * 5
    hi_c = max(hi_a, lo_c + max(width - 1, span))
    if hi_c - lo_c <= 30:
        return f"{lo_c}-{min(120, hi_c)}"
    if lo_a >= 50:
        return f"{(lo_a // 10) * 10}+"
    return "[AGE]"


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

    # Extract cluster-level PIN prefix, birth year, and stay lengths for subgroup & cohort k-checks
    profile_pin_px: dict[str, str] = {}
    profile_birth_yr: dict[str, str] = {}
    profile_stays: dict[str, set[int]] = {}
    for ck, members in sorted(cluster_units.items()):
        pins = [
            m.pin[: policy.pin_prefix_digits]
            for m in members
            if m.pin and len(m.pin) >= policy.pin_prefix_digits
        ]
        profile_pin_px[ck] = pins[0] if pins else "U"
        dobs = [m.dob[:4] for m in members if m.dob and len(m.dob) >= 4]
        profile_birth_yr[ck] = dobs[0] if dobs else "U"
        stays_ck: set[int] = set()
        adm_ds: list[dt.date] = []
        dis_ds: list[dt.date] = []
        for m in members:
            for dm in m.dates:
                if not dm.iso_date:
                    continue
                try:
                    d_obj = dt.date.fromisoformat(dm.iso_date)
                except ValueError:
                    continue
                if dm.role == "admission":
                    adm_ds.append(d_obj)
                elif dm.role == "discharge":
                    dis_ds.append(d_obj)
        for a_d in adm_ds:
            for b_d in dis_ds:
                if 0 <= (b_d - a_d).days <= 90:
                    stays_ck.add((b_d - a_d).days)
        profile_stays[ck] = stays_ck

    pin_cells: Counter[tuple[str, str, str, str]] = Counter(
        (profile_std_band[ck] or "UNK", profile_gender[ck], profile_district[ck], profile_pin_px[ck])
        for ck in cluster_units
    )
    byr_cells: Counter[tuple[str, str, str, str]] = Counter(
        (profile_std_band[ck] or "UNK", profile_gender[ck], profile_district[ck], profile_birth_yr[ck])
        for ck in cluster_units
    )

    # Decide per-cluster generalization tier under C1 (subgroup k-anonymity + cohort-aware QI suppression)
    cluster_tier: dict[str, tuple[str | None, bool, bool, str, bool]] = {}
    for ck in sorted(cluster_units):
        age = profile_age[ck]
        ages_all = [m.age for m in cluster_units[ck] if m.age is not None]
        std_band = profile_std_band[ck]
        has_rare = any(
            cond_patient_counts[cond] <= policy.rare_condition_threshold
            for cond in profile_conds[ck]
        )
        cell_k = std_cells[(std_band or "UNK", profile_gender[ck], profile_district[ck])]
        is_elderly = age is not None and age >= policy.max_age_cap

        if not policy.adaptive_widening:
            hull_b = _hull_age_band(ages_all, policy.age_band_years, policy.max_age_cap) if ages_all else std_band
            cluster_tier[ck] = (hull_b, True, True, "standard", True)
            continue

        # Balanced Hybrid: pick the truthful width<=30 age band maximizing cohort overlap across
        # matching diagnoses/stays (never '[AGE]'), and drop gender only when it merges >=1 opposite-sex match.
        band20_ck = format_age_band(age, 20, 80) if age is not None else "UNK"
        cond_cohort = min(
            (
                sum(
                    1
                    for ock in cluster_units
                    if (format_age_band(profile_age[ock], 20, 80) if profile_age[ock] is not None else "UNK") == band20_ck
                    and profile_gender[ock] == profile_gender[ck]
                    and cond in profile_conds[ock]
                )
                for cond in profile_conds[ck]
            ),
            default=99,
        )
        stay_cohort = min(
            (
                sum(
                    1
                    for ock in cluster_units
                    if (format_age_band(profile_age[ock], 20, 80) if profile_age[ock] is not None else "UNK") == band20_ck
                    and profile_gender[ock] == profile_gender[ck]
                    and st in profile_stays[ock]
                )
                for st in profile_stays[ck]
            ),
            default=99,
        )
        vuln_targeted = (
            has_rare
            or (cond_cohort <= 2)
            or (stay_cohort <= 2)
            or (cell_k < policy.min_group_size and is_elderly)
        )

        # Select best truthful wide band covering ages_all maximizing shared-trait overlap
        if ages_all:
            cands = _candidate_wide_bands(ages_all)
            best_label, best_lo, best_hi = cands[0]
            best_score = (-1, -1, "")
            for lbl, b_lo, b_hi in cands:
                cond_peers = sum(
                    1
                    for ock in cluster_units
                    if ock != ck
                    and (profile_conds[ck] & profile_conds[ock])
                    and profile_age[ock] is not None
                    and b_lo <= profile_age[ock] <= b_hi
                )
                stay_peers = sum(
                    1
                    for ock in cluster_units
                    if ock != ck
                    and (profile_stays[ck] & profile_stays[ock])
                    and profile_age[ock] is not None
                    and b_lo <= profile_age[ock] <= b_hi
                )
                total_peers = sum(
                    1
                    for ock in cluster_units
                    if profile_age[ock] is not None and b_lo <= profile_age[ock] <= b_hi
                )
                sc = (cond_peers * 4 + stay_peers * 2, total_peers, lbl)
                if sc > best_score:
                    best_score = sc
                    best_label, best_lo, best_hi = lbl, b_lo, b_hi
            band_pool_ck = best_label
            pool_lo, pool_hi = best_lo, best_hi
        else:
            band_pool_ck = "UNK"
            pool_lo, pool_hi = 0, 120

        def _in_pool(ock: str) -> bool:
            pa = profile_age[ock]
            return pa is not None and pool_lo <= pa <= pool_hi

        # Suppress gender when same-gender cohort is tiny (<=2) with opposite-sex peers, OR on singleton/rare outliers
        gender_isolates_cond = any(
            cond_patient_counts[cond] >= 2
            and (
                same_g := sum(1 for ock in cluster_units if _in_pool(ock) and cond in profile_conds[ock] and profile_gender[ock] == profile_gender[ck])
            ) <= 2
            and sum(1 for ock in cluster_units if _in_pool(ock) and cond in profile_conds[ock]) > same_g
            for cond in profile_conds[ck]
        )
        gender_isolates_stay = any(
            (
                same_g := sum(1 for ock in cluster_units if _in_pool(ock) and st in profile_stays[ock] and profile_gender[ock] == profile_gender[ck])
            ) <= 2
            and sum(1 for ock in cluster_units if _in_pool(ock) and st in profile_stays[ock]) > same_g
            for st in profile_stays[ck]
        )
        singleton_rare = (
            has_rare
            or (cell_k < policy.min_group_size and is_elderly)
            or (cell_k <= 1 and vuln_targeted)
        )
        keep_gen = not (gender_isolates_cond or gender_isolates_stay or singleton_rare)

        if has_rare or (cell_k < policy.min_group_size and is_elderly):
            wide_b = band_pool_ck if (vuln_targeted and ages_all) else (_hull_age_band(ages_all, 20, 80) if ages_all else None)
            cluster_tier[ck] = (wide_b, False, False, "suppressed", keep_gen)
        elif cell_k < policy.min_group_size or is_elderly:
            band10 = format_age_band(age, 10, 80) if age is not None else None
            k10 = wide10_cells[(band10 or "UNK", profile_gender[ck], profile_district[ck])]
            if k10 < policy.min_group_size:
                wide_b = band_pool_ck if (vuln_targeted and ages_all) else (_hull_age_band(ages_all, 20, 80) if ages_all else None)
                cluster_tier[ck] = (wide_b, False, False, "suppressed", keep_gen)
            else:
                hull10 = band_pool_ck if (vuln_targeted and ages_all) else (_hull_age_band(ages_all, 10, 80) if ages_all else band10)
                cluster_tier[ck] = (hull10, True, False, "widened", keep_gen)
        else:
            pin_ok = (profile_pin_px[ck] == "U") or (
                pin_cells[(std_band or "UNK", profile_gender[ck], profile_district[ck], profile_pin_px[ck])]
                >= policy.min_group_size
            )
            byr_ok = (profile_birth_yr[ck] == "U") or (
                byr_cells[(std_band or "UNK", profile_gender[ck], profile_district[ck], profile_birth_yr[ck])]
                >= policy.min_group_size
            )
            lvl = "standard" if byr_ok else "widened"
            hull_s = band_pool_ck if (vuln_targeted and ages_all) else (_hull_age_band(ages_all, policy.age_band_years, policy.max_age_cap) if ages_all else std_band)
            cluster_tier[ck] = (hull_s, True, pin_ok, lvl, keep_gen)

    # Post-widening joint k-anonymity & l-diversity verification across full released QIs
    if policy.adaptive_widening and cluster_tier:
        def _joint_sig(ck: str, tier: tuple[str | None, bool, bool, str, bool]) -> tuple[str, str, str, str, str]:
            t_band, t_dist, t_pin, t_lvl, t_gen = tier
            return (
                t_band or "UNK",
                profile_gender[ck] if t_gen else "U",
                profile_district[ck] if t_dist else "*",
                profile_pin_px[ck] if t_pin else "*",
                profile_birth_yr[ck] if t_lvl == "standard" else "*",
            )

        def _widest_peer_band(ck: str) -> str | None:
            ages_all_ck = [m.age for m in cluster_units[ck] if m.age is not None]
            if not ages_all_ck:
                return None
            cands = _candidate_wide_bands(ages_all_ck)
            best_lbl = cands[0][0]
            best_sc = (-1, "")
            for lbl, b_lo, b_hi in cands:
                peers = sum(
                    1
                    for ock in cluster_units
                    if profile_age[ock] is not None and b_lo <= profile_age[ock] <= b_hi
                )
                sc = (peers, lbl)
                if sc > best_sc:
                    best_sc = sc
                    best_lbl = lbl
            return best_lbl

        def _enforce_joint_k() -> None:
            min_k = policy.min_group_size
            if min_k <= 1:
                return
            # Escalate one QI tier globally across all failing classes before advancing to the next,
            # so cohorts converge at the least-destructive joint k-anonymous level.
            for step in ("drop_pin", "drop_dob", "widen_age", "drop_dist", "drop_gender"):
                sigs = {ck: _joint_sig(ck, cluster_tier[ck]) for ck in cluster_tier}
                sig_counts = Counter(sigs.values())
                failing = [ck for ck in sorted(cluster_tier) if sig_counts[sigs[ck]] < min_k]
                if not failing:
                    break
                for ck in failing:
                    band, keep_d, keep_p, lvl, keep_g = cluster_tier[ck]
                    if step == "drop_pin" and keep_p:
                        cluster_tier[ck] = (band, keep_d, False, "widened" if lvl == "standard" else lvl, keep_g)
                    elif step == "drop_dob" and lvl == "standard":
                        cluster_tier[ck] = (band, keep_d, False, "widened", keep_g)
                    elif step == "widen_age":
                        wide_lbl = _widest_peer_band(ck)
                        if wide_lbl and band != wide_lbl:
                            cluster_tier[ck] = (wide_lbl, keep_d, False, "suppressed" if not keep_d else ("widened" if lvl == "standard" else lvl), keep_g)
                    elif step == "drop_dist" and (keep_d or lvl != "suppressed"):
                        cluster_tier[ck] = (band, False, False, "suppressed", keep_g)
                    elif step == "drop_gender" and keep_g:
                        cluster_tier[ck] = (band, False, False, "suppressed", False)

        _enforce_joint_k()

        # Post-widening l-diversity check (l >= 2 distinct conditions per equivalence class when heterogeneous peers exist)
        all_corpus_conds = {cond for conds in profile_conds.values() for cond in conds}
        if len(all_corpus_conds) >= 2 and len(cluster_units) >= max(policy.min_group_size, 6):
            for _ in range(3):
                sigs = {ck: _joint_sig(ck, cluster_tier[ck]) for ck in cluster_tier}
                sig_members: dict[tuple[str, str, str, str, str], list[str]] = defaultdict(list)
                for ck, sig in sigs.items():
                    sig_members[sig].append(ck)

                l_changed = False
                for sig, members in sorted(sig_members.items()):
                    conds_cell = {cond for ck in members for cond in profile_conds[ck]}
                    if len(conds_cell) != 1:
                        continue
                    ages_cell = [m.age for ck in members for m in cluster_units[ck] if m.age is not None]
                    # Find best peer cluster carrying another diagnosis so both classes can jointly share a wider QI bracket
                    best_merge: tuple[int, tuple[str | None, bool, bool, str, bool], list[str]] | None = None
                    for ock in sorted(cluster_units):
                        if sigs[ock] == sig or not profile_conds[ock] or profile_conds[ock] <= conds_cell:
                            continue
                        peer_sig = sigs[ock]
                        peer_members = sig_members[peer_sig]
                        ages_both = ages_cell + [m.age for pm in peer_members for m in cluster_units[pm] if m.age is not None]
                        if not ages_both:
                            continue
                        shared_band = _hull_age_band(ages_both, 20, 80)
                        if shared_band == "[AGE]":
                            continue
                        same_dist = (
                            len({profile_district[x] for x in (*members, *peer_members)}) == 1
                            and all(cluster_tier[x][1] for x in (*members, *peer_members))
                        )
                        same_gen = (
                            len({profile_gender[x] for x in (*members, *peer_members)}) == 1
                            and all(cluster_tier[x][4] for x in (*members, *peer_members))
                        )
                        # Prefer keeping district & gender cleanly; cost 0 = keep both, 1 = drop dist, 2 = drop dist + gen
                        cost = (0 if same_dist else 1) + (0 if same_gen else 1)
                        cand_tier = (
                            shared_band,
                            same_dist,
                            False,
                            "widened" if same_dist else "suppressed",
                            same_gen,
                        )
                        if best_merge is None or cost < best_merge[0]:
                            best_merge = (cost, cand_tier, peer_members)
                            if cost == 0:
                                break
                    if best_merge is not None:
                        _, target_tier, peer_members = best_merge
                        for m_ck in (*members, *peer_members):
                            cluster_tier[m_ck] = target_tier
                        l_changed = True
                        break
                if not l_changed:
                    break
            _enforce_joint_k()

    # Assign collision-free pseudonyms & shifts deterministically in sorted cluster order
    cluster_pid: dict[str, str] = {}
    used_pids: set[str] = set()
    for ck in sorted(cluster_units):
        pid = make_pseudonym(ck, secret)
        attempt = 1
        while pid in used_pids:
            pid = make_pseudonym(f"{ck}#c{attempt}", secret)
            attempt += 1
        used_pids.add(pid)
        cluster_pid[ck] = pid

    # Assemble per-unit PrivacyPlan
    plans: dict[str, PrivacyPlan] = {}
    for c in clues:
        ck = cluster_map.get(c.unit_id)
        if ck:
            pid = cluster_pid[ck]
            shift = make_date_shift_days(ck, secret, policy.date_shift_max_days)
            p_band, keep_dist, keep_pin, level, keep_gen = cluster_tier[ck]
            unit_band = p_band
            if unit_band is None and c.age is not None:
                unit_band = format_age_band(c.age, 20 if level == "suppressed" else 10, 80)
            plans[c.unit_id] = PrivacyPlan(
                unit_id=c.unit_id,
                cluster_key=ck,
                pseudonym=pid,
                date_shift_days=shift,
                age_band=unit_band,
                keep_district=keep_dist,
                keep_pin_prefix=keep_pin,
                generalization_level=level,
                keep_gender=keep_gen,
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

    # 2. Shift clinical dated lines, truncate DOB to YYYY on standard tier or [DOB] when widened/suppressed
    def _line_sub(m: re.Match) -> str:
        nonlocal changed
        line = m.group(1)
        if re.search(r"(?i)\b(?:DOB|Birth|Janm)\b|जन्म", line):
            changed = True
            if plan.generalization_level != "standard":
                return re.sub(r"\b\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b", "[DOB]", line)
            iso_dob = parse_date_string(line, date_order)
            if iso_dob:
                return re.sub(r"\b\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b", iso_dob[:4], line)
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
        out_d = re.sub(r"(?i)(?:Distt?\.?|District|जिला|जनपद)\s*[:.]?\s*[^\n,।-]+", "[DISTRICT]", out)
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
                    iso_dob = parse_date_string(sv, order)
                    res[k] = iso_dob[:4] if iso_dob else "[DOB]"
                changed = True
            elif role == "date":
                order = date_orders.get((file_rel, k), "dmy")
                res[k] = shift_date_token(sv, plan.date_shift_days, order)
                changed = True
            elif role == "village":
                res[k] = "[VILLAGE]"
                changed = True
            elif role == "district":
                res[k] = sv if plan.keep_district else "[DISTRICT]"
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
                    res.pop("birthDate", None)
                else:
                    m_yr = re.match(r"^(\d{4})\b", str(res["birthDate"]).strip())
                    if m_yr:
                        res["birthDate"] = m_yr.group(1)
                    else:
                        res.pop("birthDate", None)
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
                        addr["district"] = "[DISTRICT]"
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
