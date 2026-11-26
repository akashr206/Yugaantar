"""Dynamic schema, header, JSON-key, date-order, and hidden-channel profiler."""
from __future__ import annotations

import base64
import binascii
import datetime as dt
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any

from rakshak.contracts import DateMention

# Column / key semantic roles
ROLE_PATIENT_NAME = "PATIENT_NAME"
ROLE_FIRST_NAME = "FIRST_NAME"
ROLE_LAST_NAME = "LAST_NAME"
ROLE_RELATIVE_NAME = "RELATIVE_NAME"
ROLE_RELATION = "RELATION"
ROLE_PHONE = "PHONE"
ROLE_EMAIL = "EMAIL"
ROLE_AADHAAR = "AADHAAR"
ROLE_ABHA = "ABHA"
ROLE_PAN = "PAN"
ROLE_INSURANCE_ID = "INSURANCE_ID"
ROLE_MRN = "MRN"
ROLE_DOB = "DOB"
ROLE_AGE = "AGE"
ROLE_AGE_SEX = "AGE_SEX"
ROLE_GENDER = "GENDER"
ROLE_ADDRESS = "ADDRESS"
ROLE_HOUSE = "HOUSE"
ROLE_LOCALITY = "LOCALITY"
ROLE_LANDMARK = "LANDMARK"
ROLE_VILLAGE = "VILLAGE"
ROLE_TEHSIL = "TEHSIL"
ROLE_DISTRICT = "DISTRICT"
ROLE_PIN = "PIN"
ROLE_DATE = "DATE"
ROLE_ADMISSION_DATE = "ADMISSION_DATE"
ROLE_DISCHARGE_DATE = "DISCHARGE_DATE"
ROLE_METADATA = "METADATA"
ROLE_PROTECTED_CLINICAL = "PROTECTED_CLINICAL"
ROLE_TEXT_FREE = "TEXT_FREE"
ROLE_UNKNOWN = "UNKNOWN"

_HEADER_SYNONYMS: dict[str, tuple[str, ...]] = {
    ROLE_FIRST_NAME: ("first name", "firstname", "given name", "given", "fname", "first"),
    ROLE_LAST_NAME: ("last name", "lastname", "surname", "family name", "family", "lname", "last", "maiden", "maiden name"),
    ROLE_PATIENT_NAME: (
        "patient name",
        "patients name",
        "name",
        "naam",
        "full name",
        "beneficiary name",
        "patient",
        "मरीज़ का नाम",
        "नाम",
    ),
    ROLE_RELATIVE_NAME: (
        "relative name",
        "guardian",
        "pita/pati ka naam",
        "father name",
        "husband name",
        "care of",
        "attendant",
        "next of kin",
        "raised by",
    ),
    ROLE_RELATION: ("relation", "relationship", "rel"),
    ROLE_PHONE: ("mobile", "mobile no", "phone", "phone no", "contact", "contact no", "tel", "callback", "मोबाइल"),
    ROLE_EMAIL: ("email", "e mail", "mail"),
    ROLE_AADHAAR: ("aadhaar", "aadhaar no", "aadhar", "uidai"),
    ROLE_ABHA: ("abha", "abha no", "abha id", "health id", "ndhm"),
    ROLE_PAN: ("pan", "payer pan", "pan no"),
    ROLE_INSURANCE_ID: ("pmjay id", "policy no", "insurance id", "policy number", "claim id"),
    ROLE_MRN: (
        "uhid",
        "mrn",
        "reg no",
        "patientid",
        "patient id",
        "patient no",
        "nhs no",
        "patient ref",
        "hospital uhid",
        "ipd no",
        "bill no",
        "sample id",
        "visit id",
        "ssn",
        "drivers",
        "passport",
    ),
    ROLE_DOB: ("dob", "date of birth", "birth date", "birthdate", "janm tithi", "जन्म तिथि"),
    ROLE_AGE_SEX: ("age sex", "age/sex", "age gender"),
    ROLE_AGE: ("age", "umar", "आयु"),
    ROLE_GENDER: ("gender", "sex", "ling", "लिंग"),
    ROLE_ADDRESS: ("address", "street", "line", "pata", "full address", "पता"),
    ROLE_HOUSE: ("house", "house no", "h no", "makan no"),
    ROLE_LOCALITY: ("mohalla", "locality", "ward", "colony"),
    ROLE_LANDMARK: ("landmark",),
    ROLE_VILLAGE: ("village", "village/mohalla", "gram", "gaon", "city", "town", "ग्राम"),
    ROLE_TEHSIL: ("tehsil", "taluka", "block"),
    ROLE_DISTRICT: ("district", "distt", "jila", "जिला"),
    ROLE_PIN: ("pin", "pin code", "pincode", "zip", "postal code", "postalcode"),
    ROLE_ADMISSION_DATE: ("admission date", "date of admission", "doa", "admitted on"),
    ROLE_DISCHARGE_DATE: ("discharge date", "date of discharge", "dod", "discharged on"),
    ROLE_DATE: (
        "reg date",
        "registration date",
        "registered",
        "visit date",
        "collected on",
        "reported on",
        "collected",
        "recorded date",
        "recordeddate",
        "authored on",
        "authoredon",
        "last updated",
        "lastupdated",
        "effective datetime",
        "effectivedatetime",
        "performed datetime",
        "performeddatetime",
        "start",
        "end",
        "date",
        "timestamp",
    ),
    ROLE_METADATA: ("source path", "exported by", "file path", "export path"),
    ROLE_PROTECTED_CLINICAL: (
        "doctor",
        "ref doctor",
        "referred by",
        "consultant",
        "test",
        "result",
        "value",
        "unit",
        "ref range",
        "reference range",
        "range",
        "drug",
        "strength",
        "form",
        "category",
        "schedule",
        "loinc",
        "code",
        "system",
        "display",
        "package code",
        "procedure",
        "amount inr",
        "status",
        "intent",
        "complaints",
        "history",
        "vitals",
        "diagnosis",
        "department",
        "ward",
        "hospital",
        "facility",
        "lab",
        "report type",
        "resourcetype",
        "fullurl",
        "reference",
        "profile",
        "id",
        "type",
        "use",
        "trip",
        "from",
        "to",
        "time",
        "bp",
        "pulse",
        "spo2",
    ),
    ROLE_TEXT_FREE: ("notes", "remarks", "comment", "comments", "advice", "summary"),
}

_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_B64_RX = re.compile(r"^[A-Za-z0-9+/\r\n]{24,}={0,2}$")
_DATE_TOKEN_RX = re.compile(r"(?<!\d)(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})(?!\d)")
_HEADER_PII_RX = re.compile(
    r"(?i)(?:\b(?:snj|jjh|dhh|uhid|ipd|adm)[-/]?\d+|\b[6-9]\d{9}\b|\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b)"
)

_CLEAN_FILE_HINTS = (
    "protocol",
    "formulary",
    "reference_range",
    "ward_census",
    "readme",
)


@dataclass(frozen=True)
class ColumnProfile:
    """Semantic classification and date convention for a column or JSON key."""

    raw_name: str
    role: str
    date_order: str = "dmy"
    header_has_pii: bool = False


@lru_cache(maxsize=8192)
def normalize_header(name: str) -> str:
    """Normalize a column header or JSON key for semantic matching."""
    text = unicodedata.normalize("NFKC", name)
    text = re.sub(r"\([^)]*\)", " ", text)  # drop parenthetical hints like (MM/DD/YYYY)
    text = re.sub(r"[_/.\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return text


def _fuzzy_best(norm: str, candidates: tuple[str, ...]) -> float:
    best = 0.0
    for c in candidates:
        if norm == c:
            return 1.0
        if c in norm:
            if c == "city" and not (norm.startswith("city") or re.search(r"(?<!\w)city(?!\w)", norm)):
                continue
            best = max(best, 0.88 if len(c) >= 4 else 0.70)
        elif norm in c:
            # Match whole-word sub-phrases (e.g. "last" in "last name") or substantial prefixes,
            # never accidental mid-word substrings like "end" in "attendant".
            whole_word = len(norm) >= 4 and re.search(rf"(?<!\w){re.escape(norm)}(?!\w)", c) is not None
            best = max(best, 0.88 if (whole_word or (len(norm) >= 4 and len(norm) / len(c) >= 0.50)) else 0.65)
        elif len(norm) > 3 and len(c) > 3:
            best = max(best, SequenceMatcher(None, norm, c).ratio())
    return best


_NON_PIN_QUANTITY_HINTS = (
    "income",
    "salary",
    "expense",
    "coverage",
    "amount",
    "bill",
    "cost",
    "price",
    "balance",
    "weight",
    "height",
    "count",
    "total",
    "year",
    "fips",
    "lat",
    "lon",
)


def _infer_role_from_values(values: list[str], norm_header: str = "") -> str | None:
    """Infer column role from sample values when header text is unfamiliar."""
    non_empty = [v.strip().translate(_DEVA_DIGITS) for v in values if v and v.strip()]
    if not non_empty:
        return None
    n = len(non_empty)
    if sum(1 for v in non_empty if re.fullmatch(r"(?:\+?91[\s-]?)?0?[6-9]\d{4}[\s-]?\d{5}", v)) / n >= 0.6:
        return ROLE_PHONE
    if sum(1 for v in non_empty if re.fullmatch(r"\d{4}[\s-]?\d{4}[\s-]?\d{4}", v)) / n >= 0.6:
        return ROLE_AADHAAR
    if sum(1 for v in non_empty if re.fullmatch(r"91[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}", v)) / n >= 0.6:
        return ROLE_ABHA
    if sum(1 for v in non_empty if re.fullmatch(r"[A-Z]{5}\d{4}[A-Z]", v)) / n >= 0.6:
        return ROLE_PAN
    if sum(1 for v in non_empty if re.fullmatch(r"[\w.+-]+@[\w-]+\.[\w.]+", v)) / n >= 0.6:
        return ROLE_EMAIL
    if not any(q in norm_header for q in _NON_PIN_QUANTITY_HINTS):
        if sum(1 for v in non_empty if re.fullmatch(r"[1-8]\d{5}", v)) / n >= 0.6:
            return ROLE_PIN
    if sum(1 for v in non_empty if re.fullmatch(r"\d{1,3}/[MFO]", v, re.I)) / n >= 0.6:
        return ROLE_AGE_SEX
    if sum(1 for v in non_empty if _DATE_TOKEN_RX.match(v)) / n >= 0.6:
        return ROLE_DATE
    if sum(1 for v in non_empty if re.match(r"^(?:S/o|D/o|W/o|C/o)\s+", v, re.I)) / n >= 0.5:
        return ROLE_RELATIVE_NAME
    return None


def infer_date_order_from_samples(raw_name: str, samples: list[str], hospital_hint: str = "") -> str:
    """Infer dmy / mdy / iso date convention from header hint + observed values."""
    low = raw_name.lower()
    if "mm/dd" in low or "mm-dd" in low:
        return "mdy"
    if "dd/mm" in low or "dd-mm" in low:
        return "dmy"
    if "yyyy-mm-dd" in low or "iso" in low:
        return "iso"

    dmy_votes = 0
    mdy_votes = 0
    iso_votes = 0
    for s in samples:
        if not s:
            continue
        txt = s.strip().translate(_DEVA_DIGITS)
        for m in _DATE_TOKEN_RX.finditer(txt):
            tok = m.group(1)
            parts = re.split(r"[-/.]", tok)
            if len(parts) != 3:
                continue
            if len(parts[0]) == 4:
                iso_votes += 1
                continue
            p1, p2 = int(parts[0]), int(parts[1])
            if p1 > 12:
                dmy_votes += 1
            elif p2 > 12:
                mdy_votes += 1

    if iso_votes and not (dmy_votes or mdy_votes):
        return "iso"
    if dmy_votes > mdy_votes:
        return "dmy"
    if mdy_votes > dmy_votes:
        return "mdy"
    return "dmy"


def parse_date_string(raw: str, preferred_order: str = "dmy") -> dt.date | None:
    """Parse a date string strictly honoring preferred dmy/mdy/iso order."""
    s = raw.strip().translate(_DEVA_DIGITS)
    # Strip trailing time if ISO timestamp
    if "T" in s and re.match(r"^\d{4}-\d{2}-\d{2}T", s):
        s = s.split("T", 1)[0]
    parts = re.split(r"[-/.]", s)
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return None
    a, b, c = int(parts[0]), int(parts[1]), int(parts[2])
    if len(parts[0]) == 4:
        try:
            return dt.date(a, b, c)
        except ValueError:
            return None
    year = c + (2000 if c < 100 and c <= 35 else (1900 if c < 100 else 0))
    if a > 12:
        order = "dmy"
    elif b > 12:
        order = "mdy"
    else:
        order = preferred_order
    day, month = (a, b) if order == "dmy" else (b, a)
    try:
        return dt.date(year, month, day)
    except ValueError:
        # Try flipped fallback only if preferred failed (impossible calendar date)
        day2, month2 = (b, a) if order == "dmy" else (a, b)
        try:
            return dt.date(year, month2, day2)
        except ValueError:
            return None


def header_contains_pii(header: str) -> bool:
    """Detect whether a CSV column header or JSON key itself embeds patient PII."""
    if _HEADER_PII_RX.search(header):
        return True
    # Pattern like "JJH-0077112 Zainab" or full "First Last" in a dict key
    norm = normalize_header(header)
    if norm in _HEADER_SYNONYMS.get(ROLE_PROTECTED_CLINICAL, ()):
        return False
    return False


@lru_cache(maxsize=8192)
def _classify_header_norm(norm: str) -> str:
    best_role = ROLE_UNKNOWN
    best_score = 0.0
    for role, syns in _HEADER_SYNONYMS.items():
        sc = _fuzzy_best(norm, syns)
        if sc > best_score:
            best_score = sc
            best_role = role
    return best_role if best_score >= 0.80 else ROLE_UNKNOWN


def classify_column(raw_name: str, sample_values: list[str], hospital_hint: str = "") -> ColumnProfile:
    """Classify a CSV/TSV column or JSON key using header semantics + cell values."""
    has_pii = header_contains_pii(raw_name)
    norm = normalize_header(raw_name)

    best_role = _classify_header_norm(norm)
    if best_role == ROLE_UNKNOWN:
        inferred = _infer_role_from_values(sample_values, norm_header=norm)
        if inferred is not None:
            best_role = inferred

    date_order = "dmy"
    if best_role in (ROLE_DOB, ROLE_DATE, ROLE_ADMISSION_DATE, ROLE_DISCHARGE_DATE):
        date_order = infer_date_order_from_samples(raw_name, sample_values, hospital_hint=hospital_hint)

    return ColumnProfile(
        raw_name=raw_name,
        role=best_role,
        date_order=date_order,
        header_has_pii=has_pii,
    )


def profile_table(header: list[str], body: list[list[str]], hospital_hint: str = "") -> list[ColumnProfile]:
    """Profile every column in a CSV/TSV table."""
    profiles: list[ColumnProfile] = []
    sample_rows = body[:40]
    for col_idx, h in enumerate(header):
        vals = [r[col_idx] for r in sample_rows if col_idx < len(r)]
        profiles.append(classify_column(h, vals, hospital_hint=hospital_hint))
    return profiles


def is_clean_reference_file(file_rel: str, format_type: str, parsed: Any, text: str) -> bool:
    """Detect zero-PII reference documents so clinical rules/formularies are preserved."""
    low_rel = file_rel.lower()
    if low_rel.startswith("clean/") or any(h in low_rel for h in _CLEAN_FILE_HINTS):
        # Verify no obvious direct identifiers (phone/aadhaar/abha/pan) exist inside
        flat = text.translate(_DEVA_DIGITS)
        has_id = bool(
            re.search(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)", flat)
            or re.search(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)", flat)
            or re.search(r"\b[A-Z]{5}\d{4}[A-Z]\b", flat)
        )
        if not has_id:
            return True

    if format_type in ("csv", "tsv") and isinstance(parsed, tuple) and len(parsed) == 3:
        _, header, _ = parsed
        norm_headers = {normalize_header(h) for h in header}
        pii_headers = {
            "patient name", "name", "naam", "uhid", "mrn", "mobile", "phone",
            "aadhaar", "abha", "dob", "address", "patientid",
        }
        if not (norm_headers & pii_headers) and norm_headers <= {
            "drug", "strength", "form", "category", "schedule",
            "test", "loinc", "unit", "reference range", "ref range",
        }:
            return True

    return False


def try_decode_base64_text(value: str) -> str | None:
    """Decode a base64-encoded UTF-8 text payload if valid and printable."""
    s = value.strip()
    if len(s) < 24 or not _B64_RX.match(s):
        return None
    compact = re.sub(r"\s+", "", s)
    padded = compact + "=" * (-len(compact) % 4)
    try:
        raw = base64.b64decode(padded, validate=True)
        decoded = raw.decode("utf-8", errors="strict")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    if not decoded or sum(1 for c in decoded if ord(c) < 0x20 and c not in "\t\n\r") > 0:
        return None
    return decoded


def encode_base64_text(text: str) -> str:
    """Re-encode sanitized UTF-8 text to standard base64."""
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def extract_dates_from_text(text: str, hospital_hint: str = "") -> tuple[DateMention, ...]:
    """Extract clinical/demographic DateMentions from unstructured text."""
    preferred = infer_date_order_from_samples("", [text], hospital_hint=hospital_hint)
    mentions: list[DateMention] = []
    seen: set[tuple[str, int]] = set()
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        low = line.lower()
        role = "clinical"
        if any(k in low for k in ("dob", "birth", "जन्म")):
            role = "dob"
        elif any(k in low for k in ("admission", "admitted", "भर्ती")):
            role = "admission"
        elif any(k in low for k in ("discharge", "discharged", "छुट्टी")):
            role = "discharge"
        elif any(k in low for k in ("visit", "date", "तारीख")):
            role = "visit"
        for m in _DATE_TOKEN_RX.finditer(line.translate(_DEVA_DIGITS)):
            raw_tok = m.group(1)
            # Skip obvious ranges like 12-16 or version numbers
            parts = re.split(r"[-/.]", raw_tok)
            if len(parts) == 3 and len(parts[2]) == 2 and int(parts[2]) < 20:
                continue
            parsed = parse_date_string(raw_tok, preferred_order=preferred)
            if parsed is None or not (1900 <= parsed.year <= 2035):
                continue
            key = (raw_tok, idx)
            if key not in seen:
                seen.add(key)
                mentions.append(
                    DateMention(
                        raw_text=raw_tok,
                        iso_date=parsed.isoformat(),
                        role=role,
                        date_order="iso" if len(parts[0]) == 4 else preferred,
                        field_path=f"line:{idx}",
                    )
                )
    return tuple(mentions)
