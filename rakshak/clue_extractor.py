"""Multi-format identity clue extractor and smart DMY/MDY date parser.

Extracts IdentityClues and DateMentions from RecordUnits across CSV, TSV,
nested JSON, FHIR R4 Bundles, and Devanagari/Hinglish/English clinical notes
without relying on memorized dev filenames or fixed column headers.
"""
from __future__ import annotations

import datetime as dt
import re
from collections import defaultdict
from typing import Any

from rakshak.contracts import DateMention, IdentityClues, NameClue, RecordUnit
from rakshak.translit import (
    extract_name_tokens,
    norm_aadhaar,
    norm_aadhaar_last4,
    norm_abha,
    norm_district,
    norm_location,
    norm_mrn,
    norm_phone,
    normalize_digits,
    phonetic_name_tokens,
)

_MONTH_MAP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

_ISO_DATE_RE = re.compile(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)")
_SLASH_DATE_RE = re.compile(r"\b(\d{1,2})([/.-])(\d{1,2})\2(\d{2}|\d{4})\b")
_TEXT_DATE_RE = re.compile(
    r"\b(\d{1,2})\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"(?:\s+(\d{4}))?\b",
    re.I,
)

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}")
_PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
_PMJAY_RE = re.compile(r"\b(?:PMJAY-[A-Z]{2}-\d+|POL/\d+/\d+)\b", re.I)
_ABHA_ADDR_RE = re.compile(r"\b[\w.+-]+@(?:abdm|sbx|ndhm)\b", re.I)
_AADHAAR_MASKED_RE = re.compile(r"\b[xX*]{4}[\s-]?[xX*]{4}[\s-]?\d{4}\b")
_AADHAAR_RE = re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")
_ABHA_RE = re.compile(r"(?<!\d)\d{2}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
_MRN_ANCHOR_RE = re.compile(
    r"(?im)(?:UHID|MRN|IPD(?:\s*No\.?)?|Reg\.?\s*No\.?|Patient\s*ID|पंजीकरण\s*सं\.?|आईपीडी\s*सं\.?)\s*[:.#-]?\s*([A-Z0-9][A-Z0-9/-]{4,22})"
)
_MRN_RE = re.compile(
    r"\b(?!(?:OPD|POL|PMJAY|COVID|SARS|LOINC|ICD)\b)"
    r"(?:[A-Z]{2,6}(?:\d{2,4})?[/-]\d{2,4}[/-]\d{4,8}|[A-Z]{2,6}-\d{5,9}|ADM\d{5,8})\b",
    re.I,
)
_PIN_RE = re.compile(r"(?<!\d)([1-9]\d{5})(?!\d)")

# Free-text anchors (English, Devanagari, Hinglish)
_DOC_SKIP_RE = re.compile(r"^\s*(?:Dr\.?|डॉ\.?|Consultant|Referred\s+by)\b", re.I)
_NAME_ANCHOR_RE = re.compile(
    r"(?im)(?:Patient\s*Name|Pt\.?\s*Name|मरीज़\s*का\s*नाम|नाम|Report\s+of)\s*[:\-]\s*([^\n,|/]+?)(?=\s+(?:Age|Sex|आयु|लिंग|UHID|IPD|Mob|$)|\s*[\n,|/]|$)"
)
_HINGLISH_HON_RE = re.compile(
    r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)\s+(?:ji|sahab|saheb|babu|amma|didi|behenji|bahanji|bhai)\b"
)
_KINSHIP_OPEN_RE = re.compile(
    r"(?m)^([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)\s*,\s*(?:S/[oO]|D/[oO]|W/[oO])\s+([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)"
)
_DEVA_STORY_RE = re.compile(r"([\u0900-\u097F]{2,}(?:\s+[\u0900-\u097F]{2,}){1,2})\s+को\s+\d+")
_REL_ANCHOR_RE = re.compile(
    r"(?im)(?:(?:S/[oO]|D/[oO]|W/[oO])\s*[:\-]?|पिता\s*का\s*नाम\s*[:\-]|पति\s*का\s*नाम\s*[:\-])\s*([^\n,;()]+)"
)
_ATTENDANT_RE = re.compile(
    r"(?im)\bAttendant\s*[:\-]\s*([^\n,;()]+)"
)
_CO_ANCHOR_RE = re.compile(r"\bC/[oO]\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})")
_CO_STOPWORDS = frozenset({
    "fever", "pain", "cough", "cold", "weakness", "vomiting", "headache",
    "breathlessness", "dyspnea", "chest", "abdomen", "abdominal", "loose",
    "motions", "diarrhea", "diarrhoea", "swelling", "rash", "dizziness",
    "generalised", "generalized", "acute", "chronic", "high", "low", "mild",
    "severe", "dm", "dm2", "htn", "tb", "cad", "copd", "ckd",
})

_AGE_ANCHOR_RE = re.compile(
    r"(?:Age(?:/Sex)?|आयु)\s*[:\-]\s*(\d{1,3})\s*(?:Yrs?|Y|वर्ष)?(?:\s*/\s*([MF]|पुरुष|महिला))?",
    re.I,
)
_AGE_SLASH_SEX_RE = re.compile(
    r"\b(\d{1,3})\s*(?:Yrs?|Y|वर्ष)?\s*/\s*([MF]|पुरुष|महिला)\b",
    re.I,
)
_DEVA_AGE_ONLY_RE = re.compile(r"आयु\s*[:\-]\s*(\d{1,3})\s*वर्ष")

_VILLAGE_TEXT_RE = re.compile(r"(?:Vill\.?|ग्राम)\s*[:.]?\s*([^\n,।-]+)", re.I)
_DIST_TEXT_RE = re.compile(r"(?:Distt?\.?|जिला)\s*[:.]?\s*([^\n,।-]+)", re.I)

_NON_PATIENT_CLEAN_HINTS = ("protocol", "formulary", "reference_ranges", "ward_census", "readme")


def _make_name_clue(raw: str, role: str = "patient") -> NameClue | None:
    cleaned = raw.strip(" \t\r\n.,;:-()[]{}")
    if not cleaned or len(cleaned) < 2:
        return None
    # Reject medical/numeric strings posing as names
    if re.search(r"\d", cleaned):
        return None
    toks = extract_name_tokens(cleaned)
    if not toks:
        return None
    ptoks = phonetic_name_tokens(cleaned)
    return NameClue(raw=cleaned, latin_tokens=toks, phonetic_tokens=ptoks, role=role)


def _normalize_gender(raw: str) -> str | None:
    s = normalize_digits(raw).strip().casefold()
    if s in ("m", "male", "पुरुष"):
        return "M"
    if s in ("f", "female", "महिला"):
        return "F"
    if s in ("o", "other", "transgender"):
        return "O"
    return None


def _safe_int_age(raw: Any) -> int | None:
    if isinstance(raw, int) and 0 <= raw <= 120:
        return raw
    s = normalize_digits(str(raw)).strip()
    m = re.match(r"^(\d{1,3})\b", s)
    if m:
        val = int(m.group(1))
        if 0 <= val <= 120:
            return val
    return None


def infer_file_date_orders(units: list[RecordUnit]) -> dict[tuple[str, str], str]:
    """Infer DMY vs MDY vs ISO per (file_rel, field_key_or_doc) across all units in a batch.

    Examines all slash/dash dates in the same file+column to disambiguate DMY vs MDY
    before parsing individual records.
    """
    evidence: dict[tuple[str, str], list[tuple[int, int, str, str]]] = defaultdict(list)

    def _collect(file_rel: str, field_key: str, text: str, hosp: str) -> None:
        s = normalize_digits(text)
        for m in _SLASH_DATE_RE.finditer(s):
            p1, p2 = int(m.group(1)), int(m.group(3))
            evidence[(file_rel, field_key)].append((p1, p2, hosp, field_key))

    for u in units:
        if u.structured_fields and u.format_type in ("csv", "tsv"):
            for k, v in u.structured_fields.items():
                if isinstance(v, str):
                    _collect(u.file_rel, k, v, u.hospital_hint)
        elif u.structured_fields and u.format_type == "json":
            for k, v in u.structured_fields.items():
                if isinstance(v, str):
                    _collect(u.file_rel, k, v, u.hospital_hint)
        else:
            _collect(u.file_rel, "__doc__", u.raw_text, u.hospital_hint)

    orders: dict[tuple[str, str], str] = {}
    for (file_rel, field_key), samples in evidence.items():
        low_key = field_key.lower()
        low_file = file_rel.lower()
        if "mm/dd" in low_key or "mdy" in low_key:
            orders[(file_rel, field_key)] = "mdy"
            continue
        if "dd/mm" in low_key or "dmy" in low_key:
            orders[(file_rel, field_key)] = "dmy"
            continue
        saw_p1_gt12 = any(p1 > 12 for p1, _, _, _ in samples)
        saw_p2_gt12 = any(p2 > 12 for _, p2, _, _ in samples)
        if saw_p1_gt12 and not saw_p2_gt12:
            orders[(file_rel, field_key)] = "dmy"
        elif saw_p2_gt12 and not saw_p1_gt12:
            orders[(file_rel, field_key)] = "mdy"
        else:
            hosp = samples[0][2] if samples else ""
            # Arogya clinical notes/legacy exports default MDY; billing/other hospitals default DMY
            if hosp == "ass" and "billing" not in low_file and "claims" not in low_file:
                orders[(file_rel, field_key)] = "mdy"
            else:
                orders[(file_rel, field_key)] = "dmy"
    return orders


def parse_date_string(raw: str, order: str = "dmy") -> str | None:
    """Parse a date string into YYYY-MM-DD using the inferred DMY/MDY/ISO order."""
    s = normalize_digits(raw).strip()
    m_iso = _ISO_DATE_RE.search(s)
    if m_iso:
        y, mo, d = int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))
        try:
            return dt.date(y, mo, d).isoformat()
        except ValueError:
            return None

    m_sl = _SLASH_DATE_RE.search(s)
    if m_sl:
        p1, p2, y_raw = int(m_sl.group(1)), int(m_sl.group(3)), int(m_sl.group(4))
        y = 2000 + y_raw if y_raw < 100 else y_raw
        d, mo = (p2, p1) if order == "mdy" else (p1, p2)
        if mo > 12 and d <= 12:
            d, mo = mo, d
        elif d > 31:
            return None
        try:
            return dt.date(y, mo, d).isoformat()
        except ValueError:
            # Try swapped if invalid calendar date
            try:
                return dt.date(y, d, mo).isoformat()
            except ValueError:
                return None

    m_txt = _TEXT_DATE_RE.search(s)
    if m_txt and m_txt.group(3):
        d = int(m_txt.group(1))
        mo = _MONTH_MAP[m_txt.group(2).lower()]
        y = int(m_txt.group(3))
        try:
            return dt.date(y, mo, d).isoformat()
        except ValueError:
            return None
    return None


def _classify_key(key: str) -> str:
    """Classify a CSV header or JSON key into a semantic role using token boundaries."""
    k = re.sub(r"[^a-z0-9\u0900-\u097F]+", "_", key.lower()).strip("_")
    if not k:
        return "other"
    toks = set(k.split("_"))
    if toks & {"relative", "guardian", "father", "husband", "pita", "pati"} or "pita_pati" in k:
        return "relative"
    if k in ("firstname", "first_name", "first"):
        return "first_name"
    if k in ("lastname", "last_name", "last"):
        return "last_name"
    if k in ("patient_name", "naam", "नाम", "name") or ({"patient", "name"} <= toks):
        return "patient_name"
    if toks & {"mobile", "phone", "tel", "contact", "callback", "mob", "मोबाइल"}:
        return "phone"
    if "email" in toks:
        return "email"
    if toks & {"aadhaar", "aadhar", "aadhar_no", "aadhaar_no"} or "aadhaar" in k or "aadhar" in k:
        return "aadhaar"
    if "abha" in toks and "address" in toks:
        return "abha_addr"
    if "abha" in toks:
        return "abha"
    if "pan" in toks or "payer_pan" in k:
        return "pan"
    if toks & {"pmjay", "policy", "insurance"} or "pmjay" in k or "policy" in k:
        return "insurance"
    if toks & {"uhid", "mrn", "ipd", "patientid"} or k in ("reg_no", "patient_ref", "patient_id", "hospital_uhid"):
        return "mrn"
    if toks & {"dob", "birthdate", "birth"} or "janm" in k:
        return "dob"
    if k == "age_sex":
        return "age_sex"
    if toks & {"age", "umar", "आयु"}:
        return "age"
    if toks & {"gender", "sex", "ling"}:
        return "gender"
    if toks & {"village", "gaon", "city", "ग्राम"}:
        return "village"
    if toks & {"district", "jila", "जिला"}:
        return "district"
    if toks & {"pin", "zip", "postalcode"}:
        return "pin"
    if toks & {"address", "pata", "street"}:
        return "address"
    if any(x in k for x in ("diagnos", "condition", "procedure")):
        return "condition"
    if toks & {"date", "admission", "discharge", "registered", "collected", "reported", "tithi"} or k.endswith("_date") or k.endswith("_on"):
        return "date"
    if toks & {"notes", "note", "remarks", "remark"}:
        return "free_text"
    return "other"


def _extract_address_parts(text: str, villages: list[str], districts: list[str], pins: list[str]) -> None:
    s = normalize_digits(text)
    for m in _VILLAGE_TEXT_RE.finditer(s):
        v = norm_location(m.group(1))
        if v:
            villages.append(v)
    for m in _DIST_TEXT_RE.finditer(s):
        d = norm_district(m.group(1))
        if d:
            districts.append(d)
    for m in _PIN_RE.finditer(s):
        pins.append(m.group(1))


def _scan_text_ids(
    text: str,
    phones: list[str],
    aadhaars: list[str],
    aadhaar_last4: list[str],
    abhas: list[str],
    abha_addrs: list[str],
    mrns: list[str],
    emails: list[str],
    pans: list[str],
    ins_ids: list[str],
) -> None:
    s = normalize_digits(text)
    for m in _EMAIL_RE.finditer(s):
        em = m.group(0).casefold()
        if em.endswith(("@abdm", "@sbx", "@ndhm")):
            abha_addrs.append(em)
        else:
            emails.append(em)
    for m in _ABHA_ADDR_RE.finditer(s):
        abha_addrs.append(m.group(0).casefold())
    for m in _PAN_RE.finditer(s):
        pans.append(m.group(0).upper())
    for m in _PMJAY_RE.finditer(s):
        ins_ids.append(m.group(0).upper())
    for m in _MRN_ANCHOR_RE.finditer(s):
        nm = norm_mrn(m.group(1))
        if nm:
            mrns.append(nm[0])
    for m in _MRN_RE.finditer(s):
        nm = norm_mrn(m.group(0))
        if nm:
            mrns.append(nm[0])
    # Mask matched ABHA numbers before Aadhaar/Phone regexes so trailing 12/10 digits don't bleed
    def _mask_abha(m: re.Match) -> str:
        ab = norm_abha(m.group(0))
        if ab:
            abhas.append(ab)
            return " " * len(m.group(0))
        return m.group(0)

    s_no_abha = _ABHA_RE.sub(_mask_abha, s)
    for m in _AADHAAR_MASKED_RE.finditer(s_no_abha):
        l4 = norm_aadhaar_last4(m.group(0))
        if l4:
            aadhaar_last4.append(l4)

    def _mask_aadhaar(m: re.Match) -> str:
        aa = norm_aadhaar(m.group(0))
        if aa:
            aadhaars.append(aa)
            return " " * len(m.group(0))
        return m.group(0)

    s_no_ids = _AADHAAR_RE.sub(_mask_aadhaar, s_no_abha)
    for m in _PHONE_RE.finditer(s_no_ids):
        ph = norm_phone(m.group(0))
        if ph:
            phones.append(ph)


def _extract_fhir_bundle(
    bundle: dict[str, Any],
    p_names: list[NameClue],
    r_names: list[NameClue],
    phones: list[str],
    aadhaars: list[str],
    abhas: list[str],
    mrns: list[str],
    emails: list[str],
    dobs: list[str],
    genders: list[str],
    villages: list[str],
    districts: list[str],
    pins: list[str],
    dates: list[DateMention],
    conditions: list[str],
) -> None:
    for entry in bundle.get("entry") or []:
        if not isinstance(entry, dict):
            continue
        res = entry.get("resource")
        if not isinstance(res, dict):
            continue
        rtype = res.get("resourceType")
        if rtype == "Patient":
            for nm in res.get("name") or []:
                if isinstance(nm, dict):
                    txt = nm.get("text") or " ".join(
                        list(nm.get("given") or []) + ([nm["family"]] if nm.get("family") else [])
                    )
                    nc = _make_name_clue(str(txt), "patient")
                    if nc:
                        p_names.append(nc)
            for tel in res.get("telecom") or []:
                if isinstance(tel, dict):
                    val = str(tel.get("value") or "")
                    if tel.get("system") == "email" or "@" in val:
                        emails.append(val.casefold())
                    else:
                        ph = norm_phone(val)
                        if ph:
                            phones.append(ph)
            if res.get("gender"):
                g = _normalize_gender(str(res["gender"]))
                if g:
                    genders.append(g)
            if res.get("birthDate"):
                iso = parse_date_string(str(res["birthDate"]), "iso")
                if iso:
                    dobs.append(iso)
                    dates.append(DateMention(raw_text=str(res["birthDate"]), iso_date=iso, role="dob", date_order="iso", field_path="Patient.birthDate"))
            for addr in res.get("address") or []:
                if isinstance(addr, dict):
                    if addr.get("district"):
                        districts.append(norm_district(str(addr["district"])))
                    if addr.get("city"):
                        villages.append(norm_location(str(addr["city"])))
                    if addr.get("postalCode"):
                        pins.append(normalize_digits(str(addr["postalCode"])).strip())
                    for line in addr.get("line") or []:
                        _extract_address_parts(str(line), villages, districts, pins)
            for ident in res.get("identifier") or []:
                if isinstance(ident, dict) and ident.get("value"):
                    val = str(ident["value"])
                    aa = norm_aadhaar(val)
                    ab = norm_abha(val)
                    if ab:
                        abhas.append(ab)
                    elif aa:
                        aadhaars.append(aa)
                    else:
                        nm = norm_mrn(val)
                        if nm:
                            mrns.append(nm[0])
        elif rtype == "Composition":
            subj = res.get("subject")
            if isinstance(subj, dict) and subj.get("display"):
                nc = _make_name_clue(str(subj["display"]), "patient")
                if nc:
                    p_names.append(nc)
        elif rtype == "Encounter":
            per = res.get("period")
            if isinstance(per, dict):
                if per.get("start"):
                    iso = parse_date_string(str(per["start"]), "iso")
                    if iso:
                        dates.append(DateMention(raw_text=str(per["start"]), iso_date=iso, role="admission", date_order="iso", field_path="Encounter.period.start"))
                if per.get("end"):
                    iso = parse_date_string(str(per["end"]), "iso")
                    if iso:
                        dates.append(DateMention(raw_text=str(per["end"]), iso_date=iso, role="discharge", date_order="iso", field_path="Encounter.period.end"))
        elif rtype == "Condition":
            code_obj = res.get("code")
            if isinstance(code_obj, dict):
                for c in code_obj.get("coding") or []:
                    if isinstance(c, dict) and c.get("code"):
                        conditions.append(str(c["code"]).strip())
                if code_obj.get("text"):
                    conditions.append(str(code_obj["text"]).strip().casefold())


def extract_clues_for_unit(
    unit: RecordUnit,
    date_orders: dict[tuple[str, str], str] | None = None,
) -> IdentityClues:
    """Extract normalized IdentityClues from one RecordUnit."""
    low_rel = unit.file_rel.lower()
    if any(tok in low_rel for tok in _NON_PATIENT_CLEAN_HINTS) and unit.file_rel.startswith("clean/"):
        return IdentityClues(unit_id=unit.unit_id, file_rel=unit.file_rel, loc=unit.loc)

    date_orders = date_orders or {}
    p_names: list[NameClue] = []
    r_names: list[NameClue] = []
    phones: list[str] = []
    aadhaars: list[str] = []
    aadhaar_last4: list[str] = []
    abhas: list[str] = []
    abha_addrs: list[str] = []
    mrns: list[str] = []
    emails: list[str] = []
    pans: list[str] = []
    ins_ids: list[str] = []
    dobs: list[str] = []
    ages: list[int] = []
    genders: list[str] = []
    villages: list[str] = []
    districts: list[str] = []
    pins: list[str] = []
    dates: list[DateMention] = []
    conditions: list[str] = []

    if unit.format_type == "fhir" and isinstance(unit.structured_fields, dict):
        _extract_fhir_bundle(
            unit.structured_fields,
            p_names, r_names, phones, aadhaars, abhas, mrns, emails,
            dobs, genders, villages, districts, pins, dates, conditions,
        )

    elif unit.structured_fields and unit.format_type in ("csv", "tsv", "json"):
        sf = unit.structured_fields
        # Handle split FirstName + LastName (e.g. arogya_legacy_export.csv) or nested name dict
        first_part = ""
        last_part = ""
        for k, v in sf.items():
            role = _classify_key(k)
            if role == "first_name" and isinstance(v, str):
                first_part = v.strip()
            elif role == "last_name" and isinstance(v, str):
                last_part = v.strip()
        if first_part or last_part:
            nc = _make_name_clue(f"{first_part} {last_part}".strip(), "patient")
            if nc:
                p_names.append(nc)

        # Handle nested JSON patient/name/guardian/contact/address/ids dicts (dhh_patients.json, lab reports)
        def _walk_dict(d: dict[str, Any], prefix: str = "", depth: int = 0) -> None:
            if depth > 40:
                return
            for k, v in d.items():
                if k == "_export_meta":
                    continue
                role = _classify_key(k)
                if isinstance(v, dict):
                    if k.lower() == "name" and ("first" in v or "last" in v or "local" in v):
                        lat = f"{v.get('first', '')} {v.get('last', '')}".strip()
                        for cand in (lat, str(v.get("local") or "")):
                            nc = _make_name_clue(cand, "patient")
                            if nc:
                                p_names.append(nc)
                    elif k.lower() == "guardian" and v.get("name"):
                        nc = _make_name_clue(str(v["name"]), "relative")
                        if nc:
                            r_names.append(nc)
                    else:
                        _walk_dict(v, f"{prefix}{k}.", depth + 1)
                    continue
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            if "code" in item and isinstance(item["code"], str):
                                conditions.append(item["code"].strip())
                            elif "display" in item and isinstance(item["display"], str) and role == "condition":
                                conditions.append(item["display"].strip().casefold())
                            _walk_dict(item, f"{prefix}{k}[].", depth + 1)
                        elif isinstance(item, str):
                            _scan_text_ids(item, phones, aadhaars, aadhaar_last4, abhas, abha_addrs, mrns, emails, pans, ins_ids)
                    continue

                if v is None:
                    continue
                sv = str(v).strip()
                if not sv:
                    continue

                if role == "patient_name":
                    nc = _make_name_clue(sv, "patient")
                    if nc:
                        p_names.append(nc)
                elif role == "relative":
                    cleaned_rel = re.sub(r"(?i)^(?:S/[oO]|D/[oO]|W/[oO]|C/[oO])\s*[:\-]?\s*", "", sv)
                    nc = _make_name_clue(cleaned_rel, "relative")
                    if nc:
                        r_names.append(nc)
                elif role == "phone":
                    ph = norm_phone(sv)
                    if ph:
                        phones.append(ph)
                elif role == "email":
                    emails.append(sv.casefold())
                elif role == "aadhaar":
                    aa = norm_aadhaar(sv)
                    l4 = norm_aadhaar_last4(sv)
                    if aa:
                        aadhaars.append(aa)
                    elif l4:
                        aadhaar_last4.append(l4)
                elif role == "abha":
                    ab = norm_abha(sv)
                    if ab:
                        abhas.append(ab)
                elif role == "abha_addr":
                    abha_addrs.append(sv.casefold())
                elif role == "pan":
                    if _PAN_RE.search(sv.upper()):
                        pans.append(sv.upper())
                elif role == "insurance":
                    ins_ids.append(sv.upper())
                elif role == "mrn":
                    nm = norm_mrn(sv)
                    if nm:
                        mrns.append(nm[0])
                elif role == "dob":
                    order = date_orders.get((unit.file_rel, k), "dmy")
                    iso = parse_date_string(sv, order)
                    if iso:
                        dobs.append(iso)
                        dates.append(DateMention(raw_text=sv, iso_date=iso, role="dob", date_order=order, field_path=k))
                elif role == "age":
                    a = _safe_int_age(v)
                    if a is not None:
                        ages.append(a)
                elif role == "age_sex":
                    m_as = _AGE_SLASH_SEX_RE.search(sv)
                    if m_as:
                        ages.append(int(m_as.group(1)))
                        g = _normalize_gender(m_as.group(2))
                        if g:
                            genders.append(g)
                elif role == "gender":
                    g = _normalize_gender(sv)
                    if g:
                        genders.append(g)
                elif role == "village":
                    vl = norm_location(sv)
                    if vl:
                        villages.append(vl)
                elif role == "district":
                    ds = norm_district(sv)
                    if ds:
                        districts.append(ds)
                elif role == "pin":
                    m_pin = _PIN_RE.search(normalize_digits(sv))
                    if m_pin:
                        pins.append(m_pin.group(1))
                elif role == "address":
                    _extract_address_parts(sv, villages, districts, pins)
                elif role == "condition":
                    conditions.append(sv.casefold())
                elif role == "date":
                    order = date_orders.get((unit.file_rel, k), "dmy")
                    iso = parse_date_string(sv, order)
                    if iso:
                        d_role = "admission" if "admis" in k.lower() else ("discharge" if "disch" in k.lower() else "clinical")
                        dates.append(DateMention(raw_text=sv, iso_date=iso, role=d_role, date_order=order, field_path=k))
                elif role == "free_text":
                    _extract_free_text_clues(
                        sv, date_orders.get((unit.file_rel, "__doc__"), "dmy"),
                        p_names, r_names, phones, aadhaars, aadhaar_last4, abhas, abha_addrs,
                        mrns, emails, pans, ins_ids, dobs, ages, genders, villages, districts, pins, dates, conditions,
                    )
                else:
                    # Fallback scan for hidden IDs inside unexpected keys
                    _scan_text_ids(sv, phones, aadhaars, aadhaar_last4, abhas, abha_addrs, mrns, emails, pans, ins_ids)

        _walk_dict(sf)

    # For text notes / prescriptions / discharge / single-doc JSON without structured fields
    if unit.format_type == "txt" or (unit.format_type == "json" and unit.loc == "doc" and not p_names and not mrns):
        doc_order = date_orders.get((unit.file_rel, "__doc__"), "dmy")
        _extract_free_text_clues(
            unit.raw_text, doc_order,
            p_names, r_names, phones, aadhaars, aadhaar_last4, abhas, abha_addrs,
            mrns, emails, pans, ins_ids, dobs, ages, genders, villages, districts, pins, dates, conditions,
        )

    # Only extract filename clues on single-document files (loc == "doc") — never on multi-row/multi-visit exports
    if unit.loc == "doc" and unit.metadata_hints.get("filename"):
        fname = unit.metadata_hints["filename"]
        _scan_text_ids(fname, phones, aadhaars, aadhaar_last4, abhas, abha_addrs, mrns, emails, pans, ins_ids)

    # Deduplicate preserving deterministic order
    def _dedup_names(items: list[NameClue]) -> tuple[NameClue, ...]:
        seen: set[tuple[tuple[str, ...], str]] = set()
        out: list[NameClue] = []
        for it in items:
            key = (it.latin_tokens, it.role)
            if key not in seen:
                seen.add(key)
                out.append(it)
        return tuple(out)

    def _dedup_str(items: list[str]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(x for x in items if x))

    return IdentityClues(
        unit_id=unit.unit_id,
        file_rel=unit.file_rel,
        loc=unit.loc,
        patient_names=_dedup_names(p_names),
        relative_names=_dedup_names(r_names),
        phones=_dedup_str(phones),
        aadhaars=_dedup_str(aadhaars),
        aadhaar_last4=_dedup_str(aadhaar_last4),
        abhas=_dedup_str(abhas),
        abha_addresses=_dedup_str(abha_addrs),
        mrns=_dedup_str(mrns),
        emails=_dedup_str(emails),
        pans=_dedup_str(pans),
        insurance_ids=_dedup_str(ins_ids),
        dob=dobs[0] if dobs else None,
        age=ages[0] if ages else None,
        gender=genders[0] if genders else None,
        village_norm=villages[0] if villages else None,
        district_norm=districts[0] if districts else None,
        pin=pins[0] if pins else None,
        hospital=unit.hospital_hint,
        dates=tuple(dates),
        conditions=_dedup_str(conditions),
    )


def _extract_free_text_clues(
    text: str,
    date_order: str,
    p_names: list[NameClue],
    r_names: list[NameClue],
    phones: list[str],
    aadhaars: list[str],
    aadhaar_last4: list[str],
    abhas: list[str],
    abha_addrs: list[str],
    mrns: list[str],
    emails: list[str],
    pans: list[str],
    ins_ids: list[str],
    dobs: list[str],
    ages: list[int],
    genders: list[str],
    villages: list[str],
    districts: list[str],
    pins: list[str],
    dates: list[DateMention],
    conditions: list[str] | None = None,
) -> None:
    s = normalize_digits(text)
    _scan_text_ids(s, phones, aadhaars, aadhaar_last4, abhas, abha_addrs, mrns, emails, pans, ins_ids)
    _extract_address_parts(s, villages, districts, pins)

    # Line-by-line scan skipping doctor headers for names
    for line in s.splitlines():
        if _DOC_SKIP_RE.match(line):
            continue
        for m in _NAME_ANCHOR_RE.finditer(line):
            nc = _make_name_clue(m.group(1), "patient")
            if nc:
                p_names.append(nc)
        for m in _HINGLISH_HON_RE.finditer(line):
            cand = m.group(1)
            # Exclude common non-name capitalized phrases like "Dr"
            if cand.lower() not in ("dr", "patient", "review"):
                nc = _make_name_clue(cand, "patient")
                if nc:
                    p_names.append(nc)
        for m in _KINSHIP_OPEN_RE.finditer(line):
            nc_p = _make_name_clue(m.group(1), "patient")
            nc_r = _make_name_clue(m.group(2), "relative")
            if nc_p:
                p_names.append(nc_p)
            if nc_r:
                r_names.append(nc_r)
        for m in _DEVA_STORY_RE.finditer(line):
            nc = _make_name_clue(m.group(1), "patient")
            if nc:
                p_names.append(nc)
        for m in _REL_ANCHOR_RE.finditer(line):
            rel_txt = re.split(r"(?i)\b(?:Mob|Phone|Ph|informed)\b", m.group(1))[0]
            nc = _make_name_clue(rel_txt, "relative")
            if nc:
                r_names.append(nc)
        for m in _ATTENDANT_RE.finditer(line):
            rel_txt = re.split(r"(?i)\b(?:Mob|Phone|Ph|informed)\b", m.group(1))[0]
            nc = _make_name_clue(rel_txt, "relative")
            if nc:
                r_names.append(nc)
        for m in _CO_ANCHOR_RE.finditer(line):
            cand = m.group(1).strip()
            toks = cand.lower().split()
            if toks and not any(t in _CO_STOPWORDS for t in toks):
                nc = _make_name_clue(cand, "relative")
                if nc:
                    r_names.append(nc)

        # Age / Gender parsing (explicit Age: anchor or N/M, N Yrs/F forms — never history durations)
        for m in _AGE_ANCHOR_RE.finditer(line):
            a = int(m.group(1))
            if 0 <= a <= 120:
                ages.append(a)
            if m.group(2):
                g = _normalize_gender(m.group(2))
                if g:
                    genders.append(g)
        for m in _AGE_SLASH_SEX_RE.finditer(line):
            a = int(m.group(1))
            if 0 <= a <= 120:
                ages.append(a)
            g = _normalize_gender(m.group(2))
            if g:
                genders.append(g)
        for m in _DEVA_AGE_ONLY_RE.finditer(line):
            a = int(m.group(1))
            if 0 <= a <= 120:
                ages.append(a)

        # DOB & Clinical dates
        if re.search(r"(?i)\bDOB\b|जन्म", line):
            iso = parse_date_string(line, date_order)
            if iso:
                dobs.append(iso)
                dates.append(DateMention(raw_text=line, iso_date=iso, role="dob", date_order=date_order))
        elif re.search(r"(?i)Admission|Discharge|Visit|भर्ती|छुट्टी|तारीख|Date", line):
            iso = parse_date_string(line, date_order)
            if iso:
                d_role = "admission" if re.search(r"(?i)Admission|भर्ती", line) else (
                    "discharge" if re.search(r"(?i)Discharge|छुट्टी", line) else "clinical"
                )
                dates.append(DateMention(raw_text=line, iso_date=iso, role=d_role, date_order=date_order))
        if conditions is not None and re.search(r"(?i)^(?:Diagnosis|निदान)\s*:", line):
            for icd in re.findall(r"\(([A-Z]\d{2}(?:\.\d+)?)\)", line):
                conditions.append(icd.strip())


def extract_all_clues(units: list[RecordUnit]) -> list[IdentityClues]:
    """Batch-extract IdentityClues across all RecordUnits with per-file/column date inference."""
    date_orders = infer_file_date_orders(units)
    return [extract_clues_for_unit(u, date_orders) for u in units]
