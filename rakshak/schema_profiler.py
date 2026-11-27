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
from rakshak.translit import is_valid_verhoeff

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
        "patient full name",
        "name",
        "naam",
        "full name",
        "beneficiary name",
        "subscriber name",
        "member name",
        "claimant",
        "customer name",
        "payer name",
        "patient",
        "रोगी का नाम",
        "रोगी",
        "मरीज़ का नाम",
        "नाम",
    ),
    ROLE_RELATIVE_NAME: (
        "relative name",
        "guardian",
        "pita/pati ka naam",
        "father name",
        "husband name",
        "mother name",
        "spouse name",
        "spouse",
        "father/husband",
        "care of",
        "attendant",
        "next of kin",
        "nominee",
        "nok",
        "kin",
        "raised by",
        "पिता",
        "पति",
        "अभिभावक",
    ),
    ROLE_RELATION: ("relation", "relationship", "rel"),
    ROLE_PHONE: ("mobile", "mobile no", "phone", "phone no", "contact", "contact no", "tel", "callback", "मोबाइल"),
    ROLE_EMAIL: ("email", "e mail", "mail"),
    ROLE_AADHAAR: ("aadhaar", "aadhaar no", "aadhar", "uidai"),
    ROLE_ABHA: ("abha", "abha no", "abha id", "health id", "ndhm"),
    ROLE_PAN: ("pan", "payer pan", "pan no"),
    ROLE_INSURANCE_ID: (
        "pmjay id",
        "policy no",
        "insurance id",
        "policy number",
        "claim id",
        "member id",
        "subscriber id",
        "beneficiary id",
        "card no",
        "cghs",
        "echs",
        "tpa id",
    ),
    ROLE_MRN: (
        "uhid",
        "mrn",
        "reg no",
        "reg id",
        "patientid",
        "patient id",
        "patient no",
        "nhs no",
        "patient ref",
        "hospital uhid",
        "ipd no",
        "opd no",
        "adm no",
        "admission no",
        "case id",
        "bill no",
        "sample id",
        "visit id",
        "file no",
        "health record",
        "ssn",
        "drivers",
        "passport",
    ),
    ROLE_DOB: ("dob", "date of birth", "birth date", "birthdate", "janm tithi", "जन्म तिथि"),
    ROLE_AGE_SEX: ("age sex", "age/sex", "age gender"),
    ROLE_AGE: ("age", "age yrs", "age years", "patient age", "umar", "umra", "आयु"),
    ROLE_GENDER: ("gender", "sex", "sex code", "gender code", "ling", "लिंग"),
    ROLE_ADDRESS: ("address", "street", "line", "pata", "full address", "पता"),
    ROLE_HOUSE: ("house", "house no", "h no", "makan no"),
    ROLE_LOCALITY: ("mohalla", "locality", "ward", "colony"),
    ROLE_LANDMARK: ("landmark",),
    ROLE_VILLAGE: ("village", "village/mohalla", "gram", "gaon", "city", "town", "pargana", "post office", "ग्राम"),
    ROLE_TEHSIL: ("tehsil", "taluka", "block", "mandal", "subdistrict", "sub district"),
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
_MONTH_MAP: dict[str, int] = {
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
_MONTH_TOKEN_ALT = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|"
    r"जनवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|सितम्बर|अक्टूबर|अक्तूबर|नवंबर|नवम्बर|दिसंबर|दिसम्बर"
)
_DATE_TOKEN_RX = re.compile(r"(?<!\d)(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})(?!\d)")
_TEXT_DMY_FLEX_RX = re.compile(
    rf"(?<!\d)(\d{{1,2}})(?:st|nd|rd|th)?[\s/-]+({_MONTH_TOKEN_ALT})(?:,?\s+|[\s/-]+)(\d{{2,4}})(?!\d)",
    re.I,
)
_TEXT_US_FLEX_RX = re.compile(
    rf"\b({_MONTH_TOKEN_ALT})[\s,/-]+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+|[\s/-]+)(\d{{2,4}})(?!\d)",
    re.I,
)
_TEXT_YMD_FLEX_RX = re.compile(
    rf"(?<!\d)(\d{{4}})[\s/-]+({_MONTH_TOKEN_ALT})[\s/-]+(\d{{1,2}})(?:st|nd|rd|th)?(?!\d)",
    re.I,
)
_HEADER_PII_RX = re.compile(
    r"(?i)(?:"
    r"(?<![A-Za-z0-9])(?!COVID|SARS|HIV|HBV|HCV|ICD|LOINC|SNOMED|JAN|JANUARY|FEB|FEBRUARY|MAR|MARCH|APR|APRIL|MAY|JUN|JUNE|JUL|JULY|AUG|AUGUST|SEP|SEPT|SEPTEMBER|OCT|OCTOBER|NOV|NOVEMBER|DEC|DECEMBER)[A-Za-z]{2,6}(?:\d{2,4})?[-/]\d{2,4}(?:[-/]\d{2,})+(?![A-Za-z0-9])|"
    r"(?<![A-Za-z0-9])(?!COVID|SARS|HIV|HBV|HCV|CD|IL|VIT|SG|MG|JAN|JANUARY|FEB|FEBRUARY|MAR|MARCH|APR|APRIL|MAY|JUN|JUNE|JUL|JULY|AUG|AUGUST|SEP|SEPT|SEPTEMBER|OCT|OCTOBER|NOV|NOVEMBER|DEC|DECEMBER)[A-Za-z]{2,6}[-/]\d{4,9}(?![A-Za-z0-9])|"
    r"\b(?:uhid|mrn|ipd|adm|pid|reg)[-/]?\d{3,}\b|"
    r"\b(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}\b|"
    r"\b(?!00)\d{2}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b|"
    r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b|"
    r"\b[A-Za-z]{5}\d{4}[A-Za-z]\b|"
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"
    r")"
)
_CLINICAL_TITLE_STOPWORDS = frozenset({
    "blood", "pressure", "heart", "rate", "pulse", "respiratory", "body", "temperature",
    "fasting", "random", "sugar", "glucose", "serum", "plasma", "urea", "hemoglobin",
    "haemoglobin", "creatinine", "bilirubin", "albumin", "globulin", "cholesterol",
    "triglycerides", "platelet", "white", "red", "cell", "count", "spo2", "weight",
    "height", "bmi", "systolic", "diastolic", "sodium", "potassium", "calcium", "chloride",
    "alkaline", "ferritin", "vitamin", "thyroid", "insulin", "chief", "complaint",
    "complaints", "diagnosis", "treatment", "medication", "procedure", "notes", "remarks",
    "hospital", "clinic", "department", "ward", "doctor", "consultant", "admission",
    "discharge", "registration", "visit", "report", "date", "time", "status", "total",
    "test", "lab", "sample", "specimen", "reference", "normal", "value", "unit", "range",
    "provider", "payer", "organization", "facility", "category", "description", "code",
})
_CLINICAL_CHEM_SUFFIX_RX = re.compile(
    r"(?i)(?:ase|emia|uria|osis|itis|pathy|gram|metry|sterol|globin|rubin|tinine|cose|mycin|cillin|pril|olol|statin|ectomy|otomy|plasty|scopy|lysis)$"
)
_CLINICAL_NON_GENDER_HINTS = frozenset({
    "status", "type", "category", "grade", "stage", "class", "code", "result", "flag", "test", "unit", "form", "schedule",
})
_NON_AGE_NUMERIC_HINTS = frozenset({
    "bed", "ward", "room", "floor", "count", "total", "amount", "bill", "cost", "price", "fee", "inr", "rs",
    "pulse", "bp", "spo2", "temp", "hr", "rr", "sbp", "dbp", "map", "rbs", "sugar", "hb", "hba1c",
    "wbc", "rbc", "platelet", "creatinine", "sodium", "potassium", "urea", "bilirubin", "esr", "crp", "egfr",
    "weight", "wt", "height", "ht", "bmi", "kg", "gram", "grams",
    "dose", "dosage", "mg", "ml", "mcg", "iu", "freq", "duration", "puffs", "drops", "tablets",
    "day", "days", "stay", "score", "id", "no", "num", "code", "pin", "zip", "year", "yr",
    "sr", "sno", "serial", "index", "idx", "row", "trip", "visit", "visits", "admissions", "encounters",
    "value", "result", "range", "loinc", "unit", "units", "qty", "quantity", "percent", "pct", "rate",
    "month", "months", "week", "weeks",
})
_NON_PERSON_COL_HINTS = frozenset({
    "doctor", "dr", "consultant", "physician", "referred", "author", "hospital", "clinic",
    "lab", "facility", "organization", "org", "provider", "payer", "department", "dept",
    "ward", "drug", "med", "medication", "diagnosis", "diagnoses", "condition", "complaint",
    "complaints", "symptom", "procedure", "treatment", "test", "result", "category", "status",
    "type", "remark", "remarks", "note", "notes", "comment", "comments", "summary", "advice",
    "history", "city", "town", "village", "gaon", "gram", "tehsil", "taluka", "block",
    "district", "jila", "zila", "state", "country", "address", "addr", "street", "landmark",
    "locality", "mohalla", "relation", "relationship", "from", "to", "origin", "dest",
    "trip", "package", "system", "code", "display", "unit", "range", "schedule", "form",
    "strength", "bill", "claim", "policy", "card", "panel", "specimen", "sample", "visit",
})
_NON_PERSON_CELL_WORDS = frozenset({
    "repair", "session", "surgery", "dialysis", "biopsy", "extraction", "therapy", "delivery",
    "section", "pending", "approved", "settled", "closed", "open", "finished", "active",
    "routine", "urgent", "essential", "antibiotic", "unknown", "none", "null", "na",
    "male", "female", "other", "normal", "abnormal", "positive", "negative", "high", "low",
    "mild", "moderate", "severe", "acute", "chronic", "primary", "secondary", "stable",
    "critical", "discharged", "admitted", "referred", "verified", "signed", "en", "route",
    "oxygen", "given", "tab", "cap", "inj", "syp", "drops", "oint", "medicine", "cardiology",
    "orthopaedics", "pediatrics", "paediatrics", "gynaecology", "ophthalmology", "ent",
    "fever", "pain", "cough", "infection", "disease", "syndrome", "disorder", "deficiency",
    "fracture", "failure", "diabetes", "hypertension", "typhoid", "malaria", "dengue",
}) | _CLINICAL_TITLE_STOPWORDS
_PLACE_SUFFIX_RX = re.compile(
    r"(?i)(?:nagar|pur|puram|ganj|abad|garh|kalan|khurd|tola|basti|gram|gaon|dih|pradesh)$"
)
_FLEX_PHONE_RX = re.compile(r"^(?:\(\+?91\)|\+?91|0091|0)?[\s.-]?[6-9](?:[\s.-]?\d){9}$")
_LANDLINE_RX = re.compile(r"^0\d{2,4}[-.\s]\d{6,8}$")
_FLEX_ABHA_RX = re.compile(r"^(?!00)\d{2}[-\s.]\d{4}[-\s.]\d{4}[-\s.]\d{4}$")
_UNSPACED_ABHA_RX = re.compile(r"^(?!00)\d{14}$")
_ISO_STAMP14_RX = re.compile(r"^20\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{6}$")
_FLEX_AADHAAR_RX = re.compile(r"^(?:\(\d{4}\)|\d{4})[\s.-]?\d{4}[\s.-]?\d{4}$")
_MASKED_AADHAAR_RX = re.compile(r"^(?:\([Xx*]{4}\)|[Xx*]{4})[\s.-]?[Xx*]{4}[\s.-]?\d{4}$")
_GENDER_MAP_CANON = {
    "m": "M", "male": "M", "पुरुष": "M", "पुरूष": "M", "नर": "M", "पु": "M",
    "f": "F", "female": "F", "महिला": "F", "स्त्री": "F", "मादा": "F",
    "o": "O", "other": "O", "transgender": "O", "अन्य": "O",
    "u": "U", "unknown": "U",
}
_CANON_CORE_GENDER = frozenset({"m", "f", "male", "female", "पुरुष", "पुरूष", "महिला", "स्त्री"})
_LOINC_CODE_RX = re.compile(r"^\d{3,5}-\d$")
_ICD10_CODE_RX = re.compile(r"^[A-Z]\d{2}(?:\.\d{1,2})?$")
_PMJAY_PKG_RX = re.compile(r"^[A-Z]{2}\d{3}[A-Z]$")
_ADDR_CUE_RX = re.compile(
    r"(?i)(?:\b(?:H\.?\s*No\.?|House\s*(?:No\.?)?|Makan\s*No\.?|Ward|Mohalla|Vill\.?|Village|Gram|"
    r"Tehsil|Taluka|Block|Mandal|Distt?\.?|District|Road|Rd\.?|Marg|Lane|Street|Sector|PIN)\b|"
    r"मकान|मोहल्ला|ग्राम|तहसील|जिला|जनपद|[1-8]\d{5})"
)
_CLINICAL_NOISE_RX = re.compile(
    r"(?i)(?:\b(?:mg/dL|g/dL|mmol/L|mIU/L|/uL|mmHg|Tab\.?|Cap\.?|Inj\.?|BP\s*\d|SpO2|RBS|Pulse)\b)"
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
    "mrn",
    "uhid",
    "ipd",
    "opd",
    "id",
    "code",
    "visit",
    "sample",
    "serial",
    "bed",
    "ward",
    "age",
    "stay",
    "days",
    "score",
)


def _looks_like_person_name_cell(val: str) -> bool:
    """Return True if a tabular/JSON string strongly resembles a person name rather than clinical/geo data."""
    if not val or len(val) < 2 or len(val) > 42:
        return False
    if re.search(r"[\d@:/\\=%+<>\[\]{}]", val):
        return False
    if _CLINICAL_NOISE_RX.search(val) or _ADDR_CUE_RX.search(val):
        return False
    if re.match(r"^(?:S/o|D/o|W/o|C/o)\b", val, re.I):
        return False
    words = [w.strip(".,;:'\"()-") for w in val.split() if re.search(r"[^\W_]", w)]
    if not words or len(words) > 4:
        return False
    has_deva = bool(re.search(r"[\u0900-\u097F]", val))
    for w in words:
        low = w.casefold()
        if low in _NON_PERSON_CELL_WORDS or _CLINICAL_CHEM_SUFFIX_RX.search(low):
            return False
        if _PLACE_SUFFIX_RX.search(low) and len(words) == 1:
            return False
        if not has_deva and not re.fullmatch(r"[A-Za-z][A-Za-z.'-]*", w):
            return False
        if not has_deva and not (w[0].isupper() and (len(w) == 1 or any(c.islower() for c in w[1:]) or w.isupper())):
            return False
    if has_deva:
        return sum(len(w) for w in words) >= 2
    if len(words) == 1:
        return len(words[0]) >= 3 and not _PLACE_SUFFIX_RX.search(words[0])
    # Reject 2+ word strings if final token is an Indian administrative/locality word (e.g. "Mohalla Patel", "Subhash Nagar")
    if _PLACE_SUFFIX_RX.search(words[-1].casefold()):
        return False
    return True


def _infer_role_from_values(values: list[str], norm_header: str = "") -> str | None:
    """Infer column role from sample values when header text is unfamiliar."""
    non_empty = [v.strip().translate(_DEVA_DIGITS) for v in values if v and v.strip()]
    if not non_empty:
        return None
    n = len(non_empty)
    htoks = set(norm_header.split()) if norm_header else set()

    # 1. Direct email / ABHA addresses
    if sum(1 for v in non_empty if re.fullmatch(r"[\w.+-]+@[\w-]+\.[\w.]+", v) or re.fullmatch(r"[\w.+-]+@(?:abdm|sbx|ndhm|phr|nha)", v, re.I)) / n >= 0.6:
        return ROLE_EMAIL
    # 2. ABHA (check before Aadhaar and Phone so 14 digits never bleed)
    abha_hits = sum(
        1 for v in non_empty
        if _FLEX_ABHA_RX.fullmatch(v) or (_UNSPACED_ABHA_RX.fullmatch(v) and not _ISO_STAMP14_RX.fullmatch(v))
    )
    if abha_hits / n >= 0.6:
        return ROLE_ABHA
    # 3. Aadhaar (full, masked, or bare Verhoeff-validated 12-digit)
    aadhaar_hits = sum(
        1 for v in non_empty
        if _FLEX_AADHAAR_RX.fullmatch(v) or _MASKED_AADHAAR_RX.fullmatch(v) or (len(v) == 12 and v[0] in "23456789" and is_valid_verhoeff(v))
    )
    if aadhaar_hits / n >= 0.6:
        return ROLE_AADHAAR
    # 4. Phone / Landline
    phone_hits = 0
    for v in non_empty:
        digits = re.sub(r"\D", "", v)
        if (_FLEX_PHONE_RX.fullmatch(v) and 10 <= len(digits) <= 14) or (_LANDLINE_RX.fullmatch(v) and 10 <= len(digits) <= 11):
            phone_hits += 1
    if phone_hits / n >= 0.6:
        return ROLE_PHONE
    # 5. PAN
    if sum(1 for v in non_empty if re.fullmatch(r"[A-Za-z]{5}\s?\d{4}\s?[A-Za-z]", v)) / n >= 0.6:
        return ROLE_PAN
    # 6. PIN codes (guarded against counts/IDs/amounts)
    if not any(q in norm_header for q in _NON_PIN_QUANTITY_HINTS):
        if sum(1 for v in non_empty if re.fullmatch(r"[1-8]\d{5}", v)) / n >= 0.6:
            return ROLE_PIN
    # 7. Age/Sex combined (e.g. 65/F, 48 Yrs/M)
    if sum(1 for v in non_empty if re.fullmatch(r"\d{1,3}\s*(?:yrs?|y|वर्ष)?\s*/\s*(?:male|female|[mfo]|पुरुष|महिला)", v, re.I)) / n >= 0.6:
        return ROLE_AGE_SEX
    # 8. Dates & statistical DOB vs clinical Date split
    parsed_dates: list[dt.date] = []
    for v in non_empty:
        d_obj = parse_date_string(v, preferred_order="dmy") or parse_date_string(v, preferred_order="mdy")
        if d_obj is not None and 1900 <= d_obj.year <= 2035:
            parsed_dates.append(d_obj)
    if len(parsed_dates) / n >= 0.6:
        years = sorted(d.year for d in parsed_dates)
        if any(h in htoks for h in ("dob", "birth", "janm", "bday")) or (
            len(parsed_dates) >= 2 and years[0] <= 2010 and (years[-1] <= 2012 or (years[-1] - years[0]) >= 8)
        ):
            return ROLE_DOB
        return ROLE_DATE
    # 9. Relation / Relative prefixes ("S/o Ramesh", "W/o ...")
    if sum(1 for v in non_empty if re.match(r"^(?:S/o|D/o|W/o|C/o)\s+\S+", v, re.I)) / n >= 0.5:
        return ROLE_RELATIVE_NAME
    # 10. Composite postal addresses (house/village/tehsil/PIN cues)
    if sum(1 for v in non_empty if len(_ADDR_CUE_RX.findall(v)) >= 2 and not _CLINICAL_NOISE_RX.search(v)) / n >= 0.5:
        return ROLE_ADDRESS
    # 11. Contextual Gender enum
    if not (htoks & _CLINICAL_NON_GENDER_HINTS):
        low_vals = [v.casefold() for v in non_empty]
        if all(lv in _GENDER_MAP_CANON for lv in low_vals) and any(lv in _CANON_CORE_GENDER for lv in low_vals):
            if n >= 2 or (n == 1 and low_vals[0] in ("male", "female", "पुरुष", "पुरूष", "महिला", "स्त्री", "transgender")):
                return ROLE_GENDER
    # 12. Contextual Age column (multi-row integer distribution 0..110, zero decimals)
    if n >= 2 and not (htoks & _NON_AGE_NUMERIC_HINTS) and not any("." in v for v in non_empty):
        age_vals: list[int] = []
        has_age_unit = False
        for v in non_empty:
            m_age = re.fullmatch(r"(\d{1,3})\s*(yrs?|years?|y|वर्ष|साल)?", v, re.I)
            if m_age:
                av = int(m_age.group(1))
                if 0 <= av <= 110:
                    age_vals.append(av)
                    if m_age.group(2):
                        has_age_unit = True
        if len(age_vals) / n >= 0.8 and (len(set(age_vals)) >= 2 or has_age_unit):
            return ROLE_AGE
    # 13. Unseen Hospital / Patient Record IDs (MRN)
    non_mrn_hints = _NON_PERSON_COL_HINTS | _NON_AGE_NUMERIC_HINTS | {"loinc", "package", "icd", "hsn"}
    if not (htoks & non_mrn_hints):
        mrn_hits = 0
        for v in non_empty:
            if " " in v or not (5 <= len(v) <= 24):
                continue
            if _LOINC_CODE_RX.fullmatch(v) or _ICD10_CODE_RX.fullmatch(v) or _PMJAY_PKG_RX.fullmatch(v):
                continue
            if re.fullmatch(r"(?:[A-Za-z]{1,6}(?:\d{2,4})?[-_/#]\d{3,10}(?:[-_/][A-Za-z0-9]{1,8})?|20\d{2}[-_/][A-Za-z]{2,6}[-_/]\d{3,8}|[A-Za-z]{1,4}\d{5,10})", v):
                mrn_hits += 1
            elif n >= 2 and re.fullmatch(r"0\d{6,11}", v):
                mrn_hits += 1
        if (n >= 2 and mrn_hits / n >= 0.7 and len(set(non_empty)) >= max(2, int(n * 0.4))) or (
            n == 1 and mrn_hits == 1 and re.search(r"[A-Za-z].*\d{3,}", non_empty[0])
        ):
            return ROLE_MRN
    # 14. Contextual Person Names column (multi-row unknown headers holding person names)
    if n >= 2 and not (htoks & (_NON_PERSON_COL_HINTS | _NON_AGE_NUMERIC_HINTS)):
        name_hits = sum(1 for v in non_empty if _looks_like_person_name_cell(v))
        uniq_ratio = len({v.casefold() for v in non_empty}) / n
        if name_hits / n >= 0.75 and len({v.casefold() for v in non_empty}) >= 2 and uniq_ratio >= 0.45:
            multi_hits = sum(1 for v in non_empty if len(v.split()) >= 2)
            if multi_hits / n >= 0.5 or uniq_ratio >= 0.65:
                return ROLE_PATIENT_NAME
    return None


def infer_date_order_from_samples(raw_name: str, samples: list[str], hospital_hint: str = "") -> str:
    """Infer dmy / mdy / iso date convention from header hint + observed values."""
    low = raw_name.lower()
    if "mm/dd" in low or "mm-dd" in low or "mdy" in low:
        return "mdy"
    if "dd/mm" in low or "dd-mm" in low or "dmy" in low:
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
    """Parse numeric, ISO, hyphenated/slashed month-name, ordinal, or Hindi dates."""
    s = raw.strip().translate(_DEVA_DIGITS)
    if not s:
        return None
    if "T" in s and re.match(r"^\d{4}-\d{1,2}-\d{1,2}T", s):
        s = s.split("T", 1)[0]

    def _expand_yr(y_raw: int, raw_len: int) -> int:
        if raw_len >= 4 or y_raw >= 100:
            return y_raw
        return 2000 + y_raw if y_raw <= 35 else 1900 + y_raw

    m_num = re.fullmatch(r"(\d{4})([-/.])(\d{1,2})\2(\d{1,2})|(\d{1,2})([-/.])(\d{1,2})\6(\d{2}|\d{4})", s)
    if m_num:
        if m_num.group(1) is not None:
            a, b, c = int(m_num.group(1)), int(m_num.group(3)), int(m_num.group(4))
            try:
                return dt.date(a, b, c)
            except ValueError:
                return None
        a, b, y_str = int(m_num.group(5)), int(m_num.group(7)), m_num.group(8)
        c = int(y_str)
        year = _expand_yr(c, len(y_str))
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
            day2, month2 = (b, a) if order == "dmy" else (a, b)
            try:
                return dt.date(year, month2, day2)
            except ValueError:
                return None

    m_tdmy = _TEXT_DMY_FLEX_RX.fullmatch(s)
    if m_tdmy:
        d = int(m_tdmy.group(1))
        mo = _MONTH_MAP.get(m_tdmy.group(2).casefold())
        y = _expand_yr(int(m_tdmy.group(3)), len(m_tdmy.group(3)))
        if mo:
            try:
                return dt.date(y, mo, d)
            except ValueError:
                return None

    m_tus = _TEXT_US_FLEX_RX.fullmatch(s)
    if m_tus:
        mo = _MONTH_MAP.get(m_tus.group(1).casefold())
        d = int(m_tus.group(2))
        y = _expand_yr(int(m_tus.group(3)), len(m_tus.group(3)))
        if mo:
            try:
                return dt.date(y, mo, d)
            except ValueError:
                return None

    m_tymd = _TEXT_YMD_FLEX_RX.fullmatch(s)
    if m_tymd:
        y = int(m_tymd.group(1))
        mo = _MONTH_MAP.get(m_tymd.group(2).casefold())
        d = int(m_tymd.group(3))
        if mo:
            try:
                return dt.date(y, mo, d)
            except ValueError:
                return None

    return None


def header_contains_pii(header: str) -> bool:
    """Detect whether a CSV column header or JSON key itself embeds patient PII."""
    s = header.translate(_DEVA_DIGITS).strip()
    if _HEADER_PII_RX.search(s):
        return True
    norm = normalize_header(s)
    if norm in _CLINICAL_TITLE_STOPWORDS:
        return False
    # Check against any known schema role first (handles slash-headers like "Village/Mohalla")
    if _classify_header_norm(norm) != ROLE_UNKNOWN:
        return False
    for role, syns in _HEADER_SYNONYMS.items():
        if norm in syns:
            return False
    # Devanagari multi-token header not in synonyms (likely patient/person text)
    if re.search(r"[\u0900-\u097F]", s):
        return True
    # Capitalized multi-word person name in header (e.g. "Zainab Siddiqui", "R. K. Verma")
    words = [w for w in re.split(r"[^A-Za-z.]+", s) if w]
    if len(words) >= 2 and all(w[0].isupper() for w in words):
        low_words = {w.rstrip(".").lower() for w in words}
        if not (low_words & _CLINICAL_TITLE_STOPWORDS) and not any(
            _CLINICAL_CHEM_SUFFIX_RX.search(lw) for lw in low_words
        ):
            return True
    return False


_PROTECTED_FIELD_TAILS = frozenset({
    "status", "category", "type", "amount", "inr", "cost", "price", "balance", "total",
    "package", "unit", "range", "rate", "score", "dose", "strength", "form", "schedule",
    "loinc", "code", "system", "display", "diagnosis", "diagnoses", "condition", "conditions",
    "complaint", "complaints", "history", "vitals", "procedure", "treatment", "department",
    "dept", "ward", "test", "panel", "specimen", "query", "result", "bp", "pulse", "spo2",
})
_PROTECTED_ACTOR_NOUNS = frozenset({
    "doctor", "dr", "consultant", "physician", "practitioner", "referred", "author",
    "hospital", "clinic", "facility", "organization", "org", "provider", "payer", "lab",
})
_REL_NOUNS = frozenset({
    "father", "mother", "husband", "wife", "spouse", "relative", "kin", "nok", "nominee",
    "guardian", "attendant", "caregiver", "pita", "pati", "parivar", "parijan",
    "पिता", "पति", "माता", "पत्नी", "अभिभावक", "परिजन",
})
_PERSON_SUBJECT_NOUNS = frozenset({
    "patient", "pt", "person", "subject", "beneficiary", "subscriber", "member",
    "claimant", "applicant", "holder", "client", "payee", "recipient", "resident",
    "rogi", "mariz", "naam", "नाम", "रोगी", "मरीज़",
})
_PHONE_NOUNS = frozenset({
    "phone", "mobile", "mob", "cell", "tel", "telephone", "contact", "callback",
    "helpline", "whatsapp", "ph", "मोबाइल", "फ़ोन", "फोन",
})
_ID_SUFFIX_NOUNS = frozenset({"id", "no", "num", "number", "ref", "reference", "code", "key", "uid"})
_ID_PREFIX_NOUNS = frozenset({
    "uhid", "mrn", "ipd", "opd", "adm", "admission", "reg", "regn", "registration",
    "chart", "case", "record", "episode", "file", "patient", "pt", "member", "subscriber",
    "beneficiary", "bill", "sample", "visit", "token", "ticket", "hospital", "hosp",
})
_INS_NOUNS = frozenset({"pmjay", "policy", "insurance", "claim", "cghs", "echs", "rsby", "tpa"})


def _classify_header_compositional(norm: str) -> str | None:
    """Classify unseen compound headers via token semantics before falling back to fuzzy matching."""
    toks = set(norm.split())
    if not toks:
        return None

    # Record IDs (even when prefixed with hospital/admission/visit/chart)
    if toks & {"uhid", "mrn", "ipd", "opd"} and not (toks & {"date", "dt", "time", "status"}):
        return ROLE_MRN
    if (toks & _INS_NOUNS) and (toks & _ID_SUFFIX_NOUNS or len(toks & _INS_NOUNS) == len(toks)):
        return ROLE_INSURANCE_ID
    if (toks & {"aadhaar", "aadhar", "uidai"}):
        return ROLE_AADHAAR
    if (toks & {"abha", "ndhm", "abdm"}) and "address" not in toks:
        return ROLE_ABHA
    if "pan" in toks and not (toks & {"panel", "company"}):
        return ROLE_PAN
    if (toks & _ID_PREFIX_NOUNS) and (toks & _ID_SUFFIX_NOUNS) and not (toks & {"date", "dt", "email", "mail", "phone", "mobile", "dob", "birth", "pin", "zip", "age"}):
        return ROLE_MRN

    # Dates & Birthdates (before actor/protected rules so collection_date / birth_date win)
    if toks & {"dob", "birth", "birthdate", "bday", "janm", "जन्म"}:
        return ROLE_DOB
    if toks & {"admis", "admission", "admit", "doa", "भर्ती"} and (toks & {"date", "dt", "time", "on", "तारीख", "दिनांक"} or len(toks) <= 2):
        return ROLE_ADMISSION_DATE
    if toks & {"disch", "discharge", "dod", "छुट्टी"} and (toks & {"date", "dt", "time", "on", "तारीख", "दिनांक"} or len(toks) <= 2):
        return ROLE_DISCHARGE_DATE
    if toks & {"date", "dt", "timestamp", "recordeddate", "authoredon", "registered", "collected", "reported", "तारीख", "दिनांक"}:
        return ROLE_DATE

    # Demographics (Age, Sex, Phone, Email, Addresses)
    if toks & {"email", "mail"}:
        return ROLE_EMAIL
    if (toks & _PHONE_NOUNS) and not (toks & {"email", "mail", "address", "addr"}):
        return ROLE_PHONE
    if toks & {"age", "umar", "umra", "vayas", "आयु", "उम्र"}:
        if toks & {"sex", "gender", "ling", "लिंग"}:
            return ROLE_AGE_SEX
        return ROLE_AGE
    if toks & {"sex", "gender", "ling", "लिंग"}:
        return ROLE_GENDER
    if toks & {"pin", "pincode", "zip", "postal", "postalcode", "पिन"}:
        return ROLE_PIN
    if toks & {"district", "distt", "dist", "jila", "zila", "जिला", "जनपद"}:
        return ROLE_DISTRICT
    if toks & {"tehsil", "taluk", "taluka", "block", "mandal", "subdistrict", "तहसील"}:
        return ROLE_TEHSIL
    if toks & {"village", "gaon", "gram", "city", "town", "ग्राम", "गाँव"}:
        return ROLE_VILLAGE
    if toks & {"mohalla", "locality", "colony", "मोहल्ला"}:
        return ROLE_LOCALITY
    if toks & {"house", "hno", "makan", "मकान"}:
        return ROLE_HOUSE
    if toks & {"address", "addr", "street", "pata", "पता", "residence"}:
        return ROLE_ADDRESS

    # Protect clinical, facility, billing-status, and doctor headers before name/person rules
    if (toks & _PROTECTED_FIELD_TAILS) or ((toks & _PROTECTED_ACTOR_NOUNS) and not (toks & _REL_NOUNS)):
        return ROLE_PROTECTED_CLINICAL

    # Kin / Relative vs First / Last / Patient names
    if (toks & _REL_NOUNS) or ("emergency" in toks and "contact" in toks):
        return ROLE_RELATIVE_NAME
    if toks & {"relation", "relationship"}:
        return ROLE_RELATION
    if toks & {"first", "firstname", "given", "fname", "forename"}:
        return ROLE_FIRST_NAME
    if toks & {"last", "lastname", "surname", "family", "lname", "maiden"}:
        return ROLE_LAST_NAME
    if (toks & _PERSON_SUBJECT_NOUNS) and (toks & {"name", "naam", "nm", "full", "legal", "display", "नाम"} or len(toks) == 1):
        return ROLE_PATIENT_NAME
    if "name" in toks and not (toks & (_PROTECTED_ACTOR_NOUNS | _PROTECTED_FIELD_TAILS)):
        return ROLE_PATIENT_NAME
    if toks & {"notes", "remarks", "remark", "comment", "comments", "summary", "advice", "narrative"}:
        return ROLE_TEXT_FREE
    return None


@lru_cache(maxsize=8192)
def _classify_header_norm(norm: str) -> str:
    for role, syns in _HEADER_SYNONYMS.items():
        if norm in syns:
            return role
    comp = _classify_header_compositional(norm)
    if comp is not None:
        return comp
    best_role = ROLE_UNKNOWN
    best_score = 0.0
    for role, syns in _HEADER_SYNONYMS.items():
        sc = _fuzzy_best(norm, syns)
        if sc > best_score:
            best_score = sc
            best_role = role
    return best_role if best_score >= 0.80 else ROLE_UNKNOWN


_ROLE_TO_CLUE_KIND: dict[str, str] = {
    ROLE_FIRST_NAME: "first_name",
    ROLE_LAST_NAME: "last_name",
    ROLE_PATIENT_NAME: "patient_name",
    ROLE_RELATIVE_NAME: "relative",
    ROLE_PHONE: "phone",
    ROLE_EMAIL: "email",
    ROLE_AADHAAR: "aadhaar",
    ROLE_ABHA: "abha",
    ROLE_PAN: "pan",
    ROLE_INSURANCE_ID: "insurance",
    ROLE_MRN: "mrn",
    ROLE_DOB: "dob",
    ROLE_AGE: "age",
    ROLE_AGE_SEX: "age_sex",
    ROLE_GENDER: "gender",
    ROLE_VILLAGE: "village",
    ROLE_DISTRICT: "district",
    ROLE_PIN: "pin",
    ROLE_ADDRESS: "address",
    ROLE_DATE: "date",
    ROLE_ADMISSION_DATE: "date",
    ROLE_DISCHARGE_DATE: "date",
    ROLE_TEXT_FREE: "free_text",
}


def classify_key_for_clues(key: str) -> str:
    """Unified header/key classifier shared across clue extraction and privacy layers."""
    norm = normalize_header(key)
    if not norm:
        return "other"
    toks = set(norm.split())
    if "abha" in toks and "address" in toks:
        return "abha_addr"
    if any(x in norm for x in ("diagnos", "condition", "procedure")):
        return "condition"
    role = _classify_header_norm(norm)
    return _ROLE_TO_CLUE_KIND.get(role, "other")


def clue_kind_for_role(role: str) -> str:
    """Convert a profiled ColumnProfile.role to the clue extraction kind string."""
    return _ROLE_TO_CLUE_KIND.get(role, "other")


def classify_column(raw_name: str, sample_values: list[str], hospital_hint: str = "") -> ColumnProfile:
    """Classify a CSV/TSV column or JSON key using header semantics + cell values."""
    has_pii = header_contains_pii(raw_name)
    norm = normalize_header(raw_name)

    if norm == "id":
        non_empty = [v.strip().translate(_DEVA_DIGITS) for v in sample_values if v and v.strip()]
        if non_empty and all(
            re.fullmatch(
                r"(?:urn:uuid:[0-9a-fA-F-]{36}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}|"
                r"(?:HD|DOC|EVT|OBS|ENC|COND|MED|PROC|SPEC)-\d{1,6}|[a-z]{1,4}\d{1,4}|\d{1,5})",
                v,
                re.I,
            )
            for v in non_empty
        ):
            best_role = ROLE_PROTECTED_CLINICAL
        else:
            best_role = _infer_role_from_values(sample_values, norm_header=norm) or ROLE_UNKNOWN
    else:
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


_PROTECTED_JSON_SUBTREES = frozenset({
    "_export_meta", "vitals", "diagnosis", "diagnoses", "rx", "medications",
    "panels", "facility", "hospital", "organization", "lab", "wards",
})


def profile_records_keys(records: list[dict[str, Any]], hospital_hint: str = "") -> dict[str, ColumnProfile]:
    """Pool scalar values across sibling dict records by key path and infer ColumnProfiles."""
    path_samples: dict[str, list[str]] = {}
    leaf_samples: dict[str, list[str]] = {}
    key_raw_name: dict[str, str] = {}

    for rec in records[:50]:
        if not isinstance(rec, dict):
            continue
        stack: list[tuple[str, Any]] = [("", rec)]
        while stack:
            prefix, cur = stack.pop()
            if not isinstance(cur, dict):
                continue
            for k, v in cur.items():
                if k.lower() in _PROTECTED_JSON_SUBTREES or len(prefix) > 120:
                    continue
                path = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    stack.append((path, v))
                elif isinstance(v, list):
                    for sub in v[:10]:
                        if isinstance(sub, dict):
                            stack.append((f"{path}[]", sub))
                elif v is not None:
                    sv = str(v)
                    path_samples.setdefault(path, []).append(sv)
                    leaf_samples.setdefault(k, []).append(sv)
                    key_raw_name.setdefault(path, k)
                    key_raw_name.setdefault(k, k)

    profiles: dict[str, ColumnProfile] = {}
    for key_or_path, samples in path_samples.items():
        raw_name = key_raw_name[key_or_path]
        profiles[key_or_path] = classify_column(raw_name, samples, hospital_hint=hospital_hint)
    for leaf_k, samples in leaf_samples.items():
        if leaf_k not in profiles:
            profiles[leaf_k] = classify_column(leaf_k, samples, hospital_hint=hospital_hint)
    return profiles


def is_clean_reference_file(file_rel: str, format_type: str, parsed: Any, text: str) -> bool:
    """Detect zero-PII reference documents so clinical rules/formularies are preserved."""
    low_rel = file_rel.lower()
    if low_rel.startswith("clean/") or any(h in low_rel for h in _CLEAN_FILE_HINTS):
        # Fail closed if any obvious PII/MRN/email/patient-anchor appears inside (defends against clean/ decoys)
        flat = text.translate(_DEVA_DIGITS)
        has_id = bool(
            re.search(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)", flat)
            or re.search(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)", flat)
            or re.search(r"(?<!\d)(?!00)\d{2}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}(?!\d)", flat)
            or re.search(r"\b[A-Z]{5}\d{4}[A-Z]\b", flat)
            or re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|[\w.+-]+@(?:abdm|sbx|ndhm|phr|nha|ok\w+|ybl|ibl|axl|paytm|upi)\b", flat, re.I)
            or re.search(r"(?i)\b(?:UHID|MRN|Patient\s*Name|Pt\.?\s*Name|मरीज़\s*का\s*नाम)\s*[:\-]", flat)
            or re.search(r"\b(?!(?:COVID|SARS|HIV|ICD|LOINC)\b)[A-Z]{2,6}[/-]\d{2,4}[/-]\d{4,8}\b", flat)
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
            # Skip dotted version strings (e.g. 1.2.15) unless explicitly on a dated line
            parts = re.split(r"[-/.]", raw_tok)
            if (
                len(parts) == 3
                and len(parts[2]) == 2
                and int(parts[2]) < 20
                and "." in raw_tok
                and role == "clinical"
            ):
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
