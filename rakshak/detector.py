"""Multilingual & context-aware PII detector with meaning-trap shields."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rakshak.contracts import PrivacyPlan
from rakshak.schema_profiler import (
    ROLE_AADHAAR,
    ROLE_ABHA,
    ROLE_ADDRESS,
    ROLE_ADMISSION_DATE,
    ROLE_AGE,
    ROLE_AGE_SEX,
    ROLE_DATE,
    ROLE_DISCHARGE_DATE,
    ROLE_DISTRICT,
    ROLE_DOB,
    ROLE_EMAIL,
    ROLE_FIRST_NAME,
    ROLE_GENDER,
    ROLE_HOUSE,
    ROLE_INSURANCE_ID,
    ROLE_LANDMARK,
    ROLE_LAST_NAME,
    ROLE_LOCALITY,
    ROLE_METADATA,
    ROLE_MRN,
    ROLE_PAN,
    ROLE_PATIENT_NAME,
    ROLE_PHONE,
    ROLE_PIN,
    ROLE_PROTECTED_CLINICAL,
    ROLE_RELATION,
    ROLE_RELATIVE_NAME,
    ROLE_TEHSIL,
    ROLE_VILLAGE,
    parse_date_string,
)

_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


@dataclass
class DetectionStats:
    """Counters for detected identifiers and generalized records (zero PII stored)."""

    identifiers: int = 0
    generalized: int = 0
    by_type: dict[str, int] = field(default_factory=dict)

    def add(self, kind: str, count: int = 1) -> None:
        if count <= 0:
            return
        self.identifiers += count
        self.by_type[kind] = self.by_type.get(kind, 0) + count


# --------------------------------------------------------------------------- Stage 1: Meaning-Trap Shields
_COMPLAINT_WORDS = (
    r"fever|pain|cough|cold|headache|giddiness|vomiting|loose\s+stools?|diarrh(?:o|e)a|"
    r"weakness|breathlessness|dyspnoea|chest\s+pain|abdominal|burning|itching|rash|"
    r"swelling|bleeding|dysuria|malaise|anorexia|nausea|bukhar|dard|khansi|ulti|"
    r"kamzori|chakkar|sujan"
)

_SHIELD_PATTERNS: list[re.Pattern[str]] = [
    # RFC 4122 UUIDs and urn:uuid references (FHIR identifiers/references) - never strip segments as Aadhaar/Phone
    re.compile(r"(?:urn:uuid:)?\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"),
    # Doctors (English + Devanagari) - never cross newlines!
    re.compile(r"\bDr\.[ \t]+[A-Z][a-zA-Z.'-]*(?:[ \t]+[A-Z][a-zA-Z.'-]*){0,3}(?:[ \t]*\([^)\n]{1,40}\))?"),
    re.compile(r"डॉ\.[ \t]*[^\s\n]+(?:[ \t]+[^\s\n]+){0,2}"),
    # Eponyms
    re.compile(
        r"(?i)\b(?:Addison|Parkinson|Bell|Crohn|Wilson|Hodgkin|Graves|Alzheimer|Cushing|"
        r"Hashimoto|Raynaud|Guillain|Sjogren|Kawasaki|Down|Turner|Marfan)(?:'s|s')?[ \t]+"
        r"(?:disease|syndrome|palsy|lymphoma|phenomenon)\b"
    ),
    # Clinical shorthands & common clinical phrases containing address-like words
    re.compile(r"(?i)\b(?:K/C/O|H/O)[ \t]+[^\n.;]+"),
    re.compile(rf"(?i)\bC/O[ \t]+(?:{_COMPLAINT_WORDS})\b[^\n.;]*"),
    re.compile(r"(?i)\bRoad[ \t]+traffic[ \t]+accident\b"),
    # ASHA workers & hospital wards with clinical department
    re.compile(r"(?i)\bASHA[ \t]+workers?\b"),
    re.compile(r"(?i)\bWard[ \t]+\d+[ \t]*\([^)\n]+\)(?:[ \t]*/[ \t]*Bed[ \t]*\d+)?"),
    # Hospital names
    re.compile(
        r"(?i)\b(?:Sanjeevani[ \t]+Multispeciality[ \t]+Hospital|Jeevan[ \t]+Jyoti[ \t]+Charitable[ \t]+Hospital|"
        r"Arogya[ \t]+Seva[ \t]+Sadan|District[ \t]+Hospital[ \t]+Hardoi|Lifeline[ \t]+Pathology|"
        r"Pathkind[ \t]+Labs(?:[ \t]*\([^)\n]+\))?|City[ \t]+Diagnostics|Sanjeevani[ \t]+In-house[ \t]+Lab)\b"
    ),
    # Medications, doses, frequencies, vitals, labs, ranges
    re.compile(r"(?i)\b(?:Tab\.?|Cap\.?|Inj\.?|Syp\.?|Drops|Oint\.?)[ \t]+[^\n;]+"),
    re.compile(r"(?i)\bDolo[ \t]+650(?:[ \t]+(?:OD|BD|TDS|QID))?(?:[ \t]*x[ \t]*\d+[ \t]*(?:days?|wks?|weeks?))?"),
    re.compile(r"\b[0-2]-[0-2]-[0-2]\b(?:[ \t]*x[ \t]*\d+[ \t]*(?:days?|wks?|weeks?|months?))?"),
    re.compile(r"(?i)\b\d+(?:\.\d+)?[ \t]*(?:mg/dL|g/dL|mg|mcg|ug|gm|g|mL|ml|IU|/uL|mmHg|U/L|mIU/L|°F)\b"),
    re.compile(r"(?i)\b(?:BP[ \t]*\d{2,3}/\d{2,3}|Pulse[ \t]*\d{2,3}(?:/min)?|SpO2[ \t]*\d{2,3}%|Temp[ \t]*\d{2,3}(?:\.\d+)?[ \t]*°?F)\b"),
    re.compile(
        r"(?<![A-Za-z0-9./+\-])(?![6-9]\d{4}-\d{5})(?!91-[6-9]\d{9})(?!\d{4}-\d{4}-\d{4})"
        r"\d{1,6}(?:\.\d+)?-\d{1,6}(?:\.\d+)?(?![A-Za-z0-9.\-])"
    ),
    re.compile(r"(?i)\b(?:x[ \t]*\d+[ \t]*(?:days?|wks?|weeks?|months?|yrs?)|for[ \t]+\d+[ \t]+(?:days?|wks?|weeks?|months?|yrs?))\b"),
    # ICD-10 and package codes
    re.compile(r"\b[A-TV-Z]\d{2}(?:\.\d{1,2})?\b"),
    re.compile(r"\b(?:SG|MG)\d{3}[A-Z]\b"),
]

# --------------------------------------------------------------------------- Stage 2: Direct IDs
_DIRECT_ID_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ABHA_ADDRESS", re.compile(r"[\w.+-]+@(?:abdm|sbx)\b", re.I)),
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("UPI_ID", re.compile(r"\b[\w.+-]{2,}@(?:ok(?:icici|hdfcbank|axisbank|sbi)|ybl|ibl|axl|paytm|upi)\b", re.I)),
    ("ABHA_NUMBER", re.compile(r"(?<!\d)91[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}(?!\d)")),
    ("AADHAAR_MASKED", re.compile(r"(?i)(?<![A-Za-z0-9])[X*]{4}[\s-]?[X*]{4}[\s-]?\d{4}(?!\d)")),
    ("AADHAAR", re.compile(r"(?<!\d)(?:\(\d{4}\)|\d{4})[\s-]?\d{4}[\s-]?\d{4}(?!\d)")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?91[\s-]?)?0?[6-9]\d{4}[\s-]?\d{5}(?!\d)")),
    ("PAN", re.compile(r"(?i)\b[A-Z]{5}[ \t]?\d{4}[ \t]?[A-Z]\b")),
    ("INSURANCE_ID", re.compile(r"\b(?:PMJAY-[A-Z]{2}-\d+|POL/\d+/\d+)\b")),
    (
        "MRN",
        re.compile(
            r"(?i)\b(?:SNJ/\d{2}/\d+|JJH-\d+|UHID-\d+|DHH\d{4}-\d+|DHH-IPD-\d+|"
            r"IPD-\d{4}-\d+|IP/\d{2}/\d+|ADM\d{4,}|BL/\d{2}/\d+|OPD/\d{2}/\d{2}/\d+)\b"
        ),
    ),
]

# Labeled ID fallback for unseen hospital prefixes (group 2 = ID value)
_LABELED_ID_RX = re.compile(
    r"(?i)((?:UHID|IPD\s*No\.?|Reg\.?\s*No\.?|Patient\s*No\.?|NHS\s*No\.?|MRN|Patient\s*ID|पंजीकरण\s*सं\.?|आईपीडी\s*सं\.?)[:\s]\s*)"
    r"([A-Za-z0-9/_-]{4,24}|\d{3,4}[ \t]\d{3,4}[ \t]\d{3,4})"
)

# --------------------------------------------------------------------------- Stage 3: Context-Anchored Names & Addresses
_HONORIFICS = (
    r"(?:ji|sahab|saheb|didi|bhaiya|behenji|bhai|amma|babu|chachu|chacha|chachi|"
    r"mama|mami|mausi|mausa|dadi|dada|nana|nani|kaka|kaki|bhabhi|papa|जी|साहब|दीदी|भैया)"
)

# Tuple: (entity_kind, regex, capture_group_index_to_redact)
_LABELED_NAME_PATTERNS: list[tuple[str, re.Pattern[str], int]] = [
    # Patient labels (Latin) - stop before double space or Age/Sex/UHID/IPD
    (
        "PATIENT_NAME",
        re.compile(
            r"(?im)((?:Patient\s*Name|Pt\.?\s*Name|Applicant\s*Name|Account\s*Holder|Beneficiary(?:\s*Name)?|Name)\s*:\s*)"
            r"([^\n,;|()]+?)(?=\s{2,}(?:Age|Sex|UHID|IPD|Reg|Mobile|Mob)|\s*(?:Age/Sex|Age|UHID|IPD)\s*:|\n|$)"
        ),
        2,
    ),
    # Patient labels (Devanagari)
    (
        "PATIENT_NAME",
        re.compile(
            r"(?m)((?:मरीज़\s+का\s+नाम|रोगी\s+का\s+नाम|नाम)\s*:\s*)"
            r"([^\n,;|()]+?)(?=\s{2,}|\s*(?:आयु|उम्र|लिंग|पता|मोबाइल)\s*:|\n|$)"
        ),
        2,
    ),
    # Relative labels (Devanagari)
    (
        "RELATIVE_NAME",
        re.compile(r"(?m)((?:पिता[ \t]+का[ \t]+नाम|पति[ \t]+का[ \t]+नाम|अभिभावक(?:[ \t]+का[ \t]+नाम)?)[ \t]*:[ \t]*)([^\n,;|()]+)"),
        2,
    ),
    # Relation markers (S/o, D/o, W/o, C/O <person>) - horizontal spaces only!
    (
        "RELATIVE_NAME",
        re.compile(
            r"(?m)(\b(?:S/o|D/o|W/o|C/o|S/O|D/O|W/O|C/O)[ \t]+)"
            r"([A-Z][a-zA-Z.'-]*(?:[ \t]+[A-Z][a-zA-Z.'-]*){0,3}|[\u0900-\u097F]+(?:[ \t]+[\u0900-\u097F]+){0,2})"
        ),
        2,
    ),
    # Attendant explicit label
    (
        "RELATIVE_NAME",
        re.compile(r"(?m)(Attendant:[ \t]*)([A-Z][a-zA-Z.'-]+(?:[ \t]+[A-Z][a-zA-Z.'-]+){0,3})"),
        2,
    ),
    # Possessive patient mention ("Asha's attendant")
    (
        "PATIENT_NAME",
        re.compile(r"\b([A-Z][a-zA-Z.'-]+)(?='s[ \t]+(?:attendant|relative|family|mother|father|husband|wife)\b)"),
        1,
    ),
    # WhatsApp forward sender: "[30/05/26, 9:32 pm] Sender Name:"
    (
        "PATIENT_NAME",
        re.compile(r"(?m)(^\[\d{1,2}/\d{1,2}/\d{2,4}[^\]]*\][ \t]+)([^:\n]{2,40}?)(?=:)"),
        2,
    ),
    # Fragmented note: "<Name>, W/o|S/o|D/o <Relative>"
    (
        "PATIENT_NAME",
        re.compile(r"(?m)^([A-Z][a-zA-Z.'-]+(?:[ \t]+[A-Z][a-zA-Z.'-]+){0,2})(?=[ \t]*,[ \t]*(?:S/o|D/o|W/o|C/o)\b)"),
        1,
    ),
    # FHIR Referral letter attachment: "Referral: Zainab Siddiqui (...)"
    (
        "PATIENT_NAME",
        re.compile(r"(?i)(Referral:[ \t]*)([A-Z][a-zA-Z.'-]+(?:[ \t]+[A-Z][a-zA-Z.'-]+){0,3})"),
        2,
    ),
    (
        "PATIENT_NAME",
        re.compile(r"(?m)(Report[ \t]+of[ \t]+)([A-Z][a-zA-Z.'-]+(?:[ \t]+(?:" + _HONORIFICS + r"|[A-Z][a-zA-Z.'-]+)){0,2})"),
        2,
    ),
    (
        "PATIENT_NAME",
        re.compile(r"(?i)(OPD[ \t]+note[ \t]+for[ \t]+)([A-Z][a-zA-Z.'-]+(?:[ \t]+[A-Z][a-zA-Z.'-]+){0,2})"),
        2,
    ),
    (
        "PATIENT_NAME",
        re.compile(r"(?i)((?:called[ \t]+by|witnessed[ \t]+by)[ \t]+)([A-Z][a-zA-Z.'-]+(?:[ \t]+[A-Z][a-zA-Z.'-]+){1,3})"),
        2,
    ),
    (
        "PATIENT_NAME",
        re.compile(r"(?m)(^\d{2}:\d{2}[ \t]-[ \t])([A-Z][a-zA-Z.'-]+(?:[ \t]+[A-Z][a-zA-Z.'-]+){1,3})(?=[ \t]*(?:\((?:attendant|visitor)|[ \t]*-))"),
        2,
    ),
    # Nursing note shorthand ("14:00 - Mohd. Shahid - ")
    (
        "PATIENT_NAME",
        re.compile(r"(?m)(\d{2}:\d{2}[ \t]-[ \t])([A-Z][a-zA-Z.'-]+(?:[ \t]+[A-Z][a-zA-Z.'-]+){1,3})(?=[ \t]+-)"),
        2,
    ),
    # Prefix honorifics ("Sri Ramesh Verma", "Smt. Sunita Devi", "श्री कमलेश") excluding institutional/hospital names
    (
        "PATIENT_NAME",
        re.compile(
            r"\b(?:Shri|Sri|Smt\.?|Km\.?|Mr\.?|Mrs\.?|Ms\.?|Late)[ \t]+"
            r"(?![^\n,.;]{0,50}\b(?:Hospital|Medical|Charitable|Multispeciality|Sadan|Clinic|Institute|Trust|College|University|Foundation)\b)"
            r"[A-Z][a-zA-Z.'-]+(?:[ \t]+[A-Z][a-zA-Z.'-]+){0,3}"
        ),
        0,
    ),
    (
        "PATIENT_NAME",
        re.compile(r"(?:श्री|श्रीमती|सुश्री)[ \t]+([^\s,;|()।\n]+(?:[ \t]+[^\s,;|()।\n]+){0,2})"),
        0,
    ),
    # Hinglish honorific phrases ("Shahid sahab", "Asha ji", "Reena behenji", "Ashok bhai", "Shahid babu", "Zainab amma")
    (
        "PATIENT_NAME",
        re.compile(rf"\b([A-Z][a-zA-Z.'-]+(?:[ \t]+[A-Z][a-zA-Z.'-]+)?[ \t]+(?i:{_HONORIFICS}))\b"),
        1,
    ),
    # Hinglish subject before postposition ("Ashok ka BP", "Dinesh ko bukhar") excluding generic non-name nouns
    (
        "PATIENT_NAME",
        re.compile(
            r"(?m)(?:^|(?<=\.[ \t]))(?!(?:Patient|Doctor|Dr|History|Course|Advice|Note|Result|Sample|Report|Ward|Bed|Tab|Cap|Inj)\b)"
            r"([A-Z][a-zA-Z.'-]+(?:[ \t]+[A-Z][a-zA-Z.'-]+)?)(?=[ \t]+(?:ka|ki|ke|ko|ne)[ \t]+)"
        ),
        1,
    ),
    # Hindi narrative sentence subject "<Name> को <N> दिन से"
    (
        "PATIENT_NAME",
        re.compile(r"([\u0900-\u097F]{2,}(?:[ \t]+[\u0900-\u097F]{2,}){1,2})(?=[ \t]+को[ \t]+\d+[ \t]+दिन[ \t]+से)"),
        1,
    ),
]

_ADDRESS_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("HOUSE", re.compile(r"(?im)\b(?:H\.?[ \t]*No\.?|House(?:[ \t]+No\.?)?|Makan[ \t]+No\.?)[: \t]*[A-Za-z0-9/-]+")),
    ("HOUSE", re.compile(r"(?m)मकान[ \t]+नं\.?[: \t]*\d+[A-Za-z0-9/-]*")),
    ("LOCALITY", re.compile(r"(?im)\bWard[ \t]+\d+\b(?![ \t]*\()")),
    ("LOCALITY", re.compile(r"(?im)\bMohalla[: \t]+[^,\n-]+?(?=,|[ \t]+(?:near|opp|behind|Vill|Gram|Tehsil|Distt)|$)")),
    ("LOCALITY", re.compile(r"(?m)मोहल्ला[: \t]+[^,\n-]+?(?=,|[ \t]+(?:तहसील|जिला|ग्राम|के[ \t]+पास|के[ \t]+पीछे|के[ \t]+सामने)|$)")),
    ("LANDMARK", re.compile(r"(?im)\b(?:near|opp\.?|opposite|behind)[ \t]+[^,\n-]+?(?=,|[ \t]+(?:Vill|Gram|Tehsil|Distt)|$)")),
    ("LANDMARK", re.compile(r"(?m)[^\s,।]+(?:[ \t]+[^\s,।]+){0,3}[ \t]+के[ \t]+(?:पास|पीछे|सामने)")),
    ("VILLAGE", re.compile(r"(?im)\b(?:Vill\.?|Village|Gram)[: \t]+[^,\n-]+?(?=,|[ \t]+(?:Tehsil|Distt|PIN)|$)")),
    ("VILLAGE", re.compile(r"(?m)ग्राम[: \t]+[^,\n-]+?(?=,|[ \t]+(?:तहसील|जिला)|$)")),
    ("TEHSIL", re.compile(r"(?im)\bTehsil[: \t]+[^,\n-]+?(?=,|[ \t]+(?:Distt|PIN)|$)")),
    ("TEHSIL", re.compile(r"(?m)तहसील[: \t]+[^,\n-]+?(?=,|[ \t]+जिला|$)")),
    # Street / lane / road / marg / gali / sector addresses (case-sensitive capitalized words, guarding clinical/hospital terms)
    (
        "LOCALITY",
        re.compile(
            r"(?m)\b(?:(?i:Lane|Street|St\.?|Sector|Phase|Block|Gali)[ \t]+[A-Za-z0-9][A-Za-z0-9 .'-]{1,30}|"
            r"(?!(?:Hospital|Medical|Clinic|College|Sanjeevani|Arogya|Jeevan|District)\b)"
            r"[A-Z][A-Za-z0-9.'-]*(?:[ \t]+[A-Z][A-Za-z0-9.'-]*){0,2}[ \t]+(?:Road|Rd\.?|Marg|Lane|Street|St\.?|Gali|Chowk|Nagar|Vihar|Enclave|Colony))\b"
        ),
    ),
]

_PIN_IN_ADDRESS_RX = re.compile(r"(?:-\s|PIN[:\s]\s*|Distt\.\s[A-Za-z\s]{3,25})(\b[1-8]\d{5}\b)")
_STANDALONE_PIN_RX = re.compile(r"(?:^|[-\s,])([1-8]\d{5})(?=\s*$|\s*[,\n])", re.M)

_FREE_AGE_EN_RX = re.compile(r"(?i)(\bAge/Sex:\s*|\bAge:\s*)(\d{1,3})(\s*(?:Yrs?|Years?)?(?:/[MFO])?)\b")
_FREE_AGE_WA_RX = re.compile(r"(?i)(Report of \[PATIENT_NAME\]\s+)(\d{1,3})(\s*Y\s*/\s*[MFO])")
_FREE_AGE_HI_RX = re.compile(r"(आयु:\s*)(\d{1,3})(\s*वर्ष(?:\s*/\s*(?:पुरुष|महिला|अन्य))?)")


def _shield_text(text: str) -> tuple[str, list[str]]:
    """Replace clinical meaning-trap spans with temporary placeholders."""
    shields: list[str] = []

    def _repl(m: re.Match[str]) -> str:
        idx = len(shields)
        shields.append(m.group(0))
        return f"\x00SHIELD_{idx}\x00"

    protected = text
    for rx in _SHIELD_PATTERNS:
        protected = rx.sub(_repl, protected)
    return protected, shields


def _unshield_text(text: str, shields: list[str]) -> str:
    """Restore shielded clinical spans back into text."""
    for idx in range(len(shields) - 1, -1, -1):
        text = text.replace(f"\x00SHIELD_{idx}\x00", shields[idx])
    return text


_COMMA_VILLAGE_RX = re.compile(r"(?m),[ \t]*([^,\n\[\]]{3,35}?)(?=[ \t]*,[ \t]*\[TEHSIL\])")
_SUPPRESS_DIST_EN_RX = re.compile(r"(?i)\bDistt?\.?[ \t]+[A-Za-z]+(?:[ \t]+[A-Za-z]+){0,2}(?=[ \t]*(?:,|-|\n|$))")
_SUPPRESS_DIST_HI_RX = re.compile(r"जिला[ \t]+(?!(?:अस्पताल|चिकित्सालय)\b)[\u0900-\u097F]+(?:[ \t]+[\u0900-\u097F]+){0,1}")
_FAST_SAFE_TOKEN_RX = re.compile(r"^[A-Za-z_-]{1,36}$")


def _apply_shadow_sub(
    text: str,
    rx: re.Pattern[str],
    kind: str,
    stats: DetectionStats,
    replacement_fn=None,
    group_idx: int = 0,
) -> str:
    """Run regex on 1:1 Devanagari-digit-normalized shadow string and splice original."""
    shadow = text if text.isascii() else text.translate(_DEVA_DIGITS)
    matches = list(rx.finditer(shadow))
    if not matches:
        return text
    out: list[str] = []
    pos = 0
    for m in matches:
        s, e = m.span(group_idx)
        if s < pos or s == e:
            continue
        raw_sub = text[s:e]
        if "\x00SHIELD_" in raw_sub or f"[{kind}]" in raw_sub:
            continue
        out.append(text[pos:s])
        rep = replacement_fn(m, raw_sub) if replacement_fn else f"[{kind}]"
        out.append(rep)
        pos = e
        stats.add(kind)
    out.append(text[pos:])
    return "".join(out)


def format_age_band(age: int, band_years: int = 5, max_cap: int = 85) -> str:
    """Compute a truthful age band containing `age`."""
    if age >= max_cap:
        return f"{max_cap}+"
    lo = (age // band_years) * band_years
    hi = min(lo + band_years - 1, max_cap - 1)
    return f"{lo}-{hi}"


def clean_free_text(
    text: str,
    stats: DetectionStats,
    plan: PrivacyPlan | None = None,
    clean_only_direct_ids: bool = False,
    hospital_hint: str = "",
    preferred_order: str | None = None,
) -> str:
    """Detect and redact PII in free text while protecting clinical meaning traps."""
    if not text or _FAST_SAFE_TOKEN_RX.fullmatch(text):
        return text

    if clean_only_direct_ids:
        res, direct_shields = _shield_text(text)
        for kind, rx in _DIRECT_ID_PATTERNS[:9]:  # email, upi, abha, aadhaar, phone, pan, insurance
            res = _apply_shadow_sub(res, rx, kind, stats)
        return _unshield_text(res, direct_shields)

    # 1. Shield protected clinical expressions first
    work, shields = _shield_text(text)

    # 2. Direct structured IDs (shadow-matched across ASCII & Devanagari digits)
    for kind, rx in _DIRECT_ID_PATTERNS:
        work = _apply_shadow_sub(work, rx, kind, stats)

    # Labeled unseen-format MRNs
    work = _apply_shadow_sub(work, _LABELED_ID_RX, "MRN", stats, group_idx=2)

    # 3. Context-anchored patient & relative names
    for kind, rx, grp in _LABELED_NAME_PATTERNS:
        def _name_rep(m: re.Match[str], raw_sub: str, k: str = kind) -> str:
            if raw_sub.strip().startswith("[") and raw_sub.strip().endswith("]"):
                return raw_sub
            return f"[{k}]"
        work = _apply_shadow_sub(work, rx, kind, stats, replacement_fn=_name_rep, group_idx=grp)

    # 4. Address components (House, Locality, Landmark, Village, Tehsil, PIN)
    for kind, rx in _ADDRESS_PATTERNS:
        work = _apply_shadow_sub(work, rx, kind, stats)

    # Unprefixed village sitting directly before ", [TEHSIL]" in Address lines
    work = _apply_shadow_sub(
        work,
        _COMMA_VILLAGE_RX,
        "VILLAGE",
        stats,
        group_idx=1,
    )

    def _pin_rep(m: re.Match[str], raw_sub: str) -> str:
        digits = raw_sub.translate(_DEVA_DIGITS)
        if plan and not plan.keep_pin_prefix:
            return "[PIN]"
        return f"{digits[:3]}***"

    work = _apply_shadow_sub(work, _PIN_IN_ADDRESS_RX, "PIN", stats, replacement_fn=_pin_rep, group_idx=1)
    work = _apply_shadow_sub(work, _STANDALONE_PIN_RX, "PIN", stats, replacement_fn=_pin_rep, group_idx=1)

    # 5. Shift clinical dates, generalize ages, and suppress district/DOB when required by PrivacyPlan
    if plan is not None:
        if not plan.keep_district:
            work = _SUPPRESS_DIST_EN_RX.sub("[DISTRICT]", work)
            work = _SUPPRESS_DIST_HI_RX.sub("[DISTRICT]", work)
        work = shift_dates_in_text(
            work,
            plan.date_shift_days,
            hospital_hint=hospital_hint,
            is_suppressed=(plan.generalization_level != "standard"),
            preferred_order=preferred_order,
        )
        work = _shift_month_name_dates(work, plan.date_shift_days)
        work = generalize_ages_in_text(work, plan)

    # 6. Restore protected clinical spans
    work = _unshield_text(work, shields)
    return work


def shift_dates_in_text(
    text: str,
    shift_days: int,
    hospital_hint: str = "",
    is_suppressed: bool = False,
    preferred_order: str | None = None,
) -> str:
    """Shift date tokens inside free text while preserving original formatting style."""
    date_rx = re.compile(r"(?<!\d)(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})(?!\d)")
    preferred = preferred_order if preferred_order in ("dmy", "mdy") else "dmy"
    # Infer file-level order quickly
    dmy_v = mdy_v = 0
    shadow = text.translate(_DEVA_DIGITS)
    for m in date_rx.finditer(shadow):
        parts = re.split(r"[-/.]", m.group(1))
        if len(parts) == 3 and len(parts[0]) != 4:
            p1, p2 = int(parts[0]), int(parts[1])
            if p1 > 12:
                dmy_v += 1
            elif p2 > 12:
                mdy_v += 1
    if dmy_v > mdy_v:
        preferred = "dmy"
    elif mdy_v > dmy_v:
        preferred = "mdy"

    def _repl(m: re.Match[str]) -> str:
        tok = m.group(1)
        parts = re.split(r"[-/.]", tok)
        if len(parts) != 3:
            return tok
        # Guard ambiguous non-ISO 2-digit year tokens with year < 20 (skip YYYY-MM-DD where parts[2] is day 01..19)
        if len(parts[0]) != 4 and len(parts[2]) == 2 and int(parts[2]) < 20:
            return tok
        parsed = parse_date_string(tok, preferred_order=preferred)
        if parsed is None or not (1900 <= parsed.year <= 2035):
            return tok
        # Check line context for DOB
        line_start = shadow.rfind("\n", 0, m.start()) + 1
        line_prefix = shadow[line_start:m.start()].lower()
        if is_suppressed and any(k in line_prefix for k in ("dob", "birth", "जन्म")):
            return "[DOB]"
        shifted = parsed + __import__("datetime").timedelta(days=shift_days)
        sep = "-" if "-" in tok else ("." if "." in tok else "/")
        if len(parts[0]) == 4:
            return f"{shifted.year:04d}{sep}{shifted.month:02d}{sep}{shifted.day:02d}"
        yr = f"{shifted.year % 100:02d}" if len(parts[2]) == 2 else f"{shifted.year:04d}"
        order = "dmy" if int(parts[0]) > 12 else ("mdy" if int(parts[1]) > 12 else preferred)
        if order == "dmy":
            return f"{shifted.day:02d}{sep}{shifted.month:02d}{sep}{yr}"
        return f"{shifted.month:02d}{sep}{shifted.day:02d}{sep}{yr}"

    return date_rx.sub(_repl, shadow)


def _shift_month_name_dates(text: str, shift_days: int) -> str:
    """Shift textual dates such as '5 September' cleanly using privacy.shift_date_token."""
    from rakshak.privacy import _TEXT_MONTH_DATE_RE, shift_date_token

    return _TEXT_MONTH_DATE_RE.sub(lambda m: shift_date_token(m.group(0), shift_days), text)


def generalize_ages_in_text(text: str, plan: PrivacyPlan) -> str:
    """Replace age integers in age markers with truthful PrivacyPlan age bands."""
    def _band_for(age_val: int) -> str:
        if plan.age_band:
            return plan.age_band
        if plan.generalization_level == "suppressed":
            return "[AGE]"
        return format_age_band(age_val)

    def _rep_en(m: re.Match[str]) -> str:
        age_val = int(m.group(2).translate(_DEVA_DIGITS))
        sfx = m.group(3)
        if not getattr(plan, "keep_gender", True):
            sfx = re.sub(r"/[MFO]$", "/[GENDER]", sfx, flags=re.I)
        return f"{m.group(1)}{_band_for(age_val)}{sfx}"

    def _rep_wa(m: re.Match[str]) -> str:
        age_val = int(m.group(2).translate(_DEVA_DIGITS))
        sfx = m.group(3)
        if not getattr(plan, "keep_gender", True):
            sfx = re.sub(r"/\s*[MFO]$", "/[GENDER]", sfx, flags=re.I)
        return f"{m.group(1)}{_band_for(age_val)}{sfx}"

    def _rep_hi(m: re.Match[str]) -> str:
        age_val = int(m.group(2).translate(_DEVA_DIGITS))
        sfx = m.group(3)
        if not getattr(plan, "keep_gender", True):
            sfx = re.sub(r"\s*/\s*(?:पुरुष|महिला|अन्य)$", " / [GENDER]", sfx)
        return f"{m.group(1)}{_band_for(age_val)}{sfx}"

    out = _FREE_AGE_EN_RX.sub(_rep_en, text)
    out = _FREE_AGE_WA_RX.sub(_rep_wa, out)
    out = _FREE_AGE_HI_RX.sub(_rep_hi, out)
    return out


def sanitize_structured_cell(
    value: str,
    role: str,
    date_order: str,
    plan: PrivacyPlan | None,
    stats: DetectionStats,
    hospital_hint: str = "",
) -> str:
    """Sanitize a single CSV/TSV cell or JSON scalar string using its profiled column/key role."""
    if not value or not value.strip():
        return value

    norm_val = value.strip()
    is_suppressed = plan is not None and plan.generalization_level == "suppressed"

    if role in (ROLE_PATIENT_NAME, ROLE_FIRST_NAME, ROLE_LAST_NAME):
        stats.add("PATIENT_NAME")
        return "[PATIENT_NAME]"
    if role == ROLE_RELATIVE_NAME:
        stats.add("RELATIVE_NAME")
        return "[RELATIVE_NAME]"
    if role == ROLE_RELATION:
        return value  # S/o, D/o, W/o alone is non-identifying structural relation
    if role == ROLE_PHONE:
        stats.add("PHONE")
        return "[PHONE]"
    if role == ROLE_EMAIL:
        stats.add("EMAIL")
        return "[EMAIL]"
    if role == ROLE_AADHAAR:
        if "x" in norm_val.lower() or "*" in norm_val:
            stats.add("AADHAAR_MASKED")
            return "[AADHAAR_MASKED]"
        stats.add("AADHAAR")
        return "[AADHAAR]"
    if role == ROLE_ABHA:
        stats.add("ABHA_NUMBER")
        return "[ABHA_NUMBER]"
    if role == ROLE_PAN:
        stats.add("PAN")
        return "[PAN]"
    if role == ROLE_INSURANCE_ID:
        stats.add("INSURANCE_ID")
        return "[INSURANCE_ID]"
    if role == ROLE_MRN:
        # Inject linked pseudonym into primary hospital/patient ID cells (including unseen chaos prefixes)
        stats.add("MRN")
        if plan and plan.pseudonym and re.search(r"[A-Za-z]{2,8}[/-]?\d{3,}", norm_val):
            return plan.pseudonym
        return "[MRN]"
    if role == ROLE_HOUSE:
        stats.add("HOUSE")
        return "[HOUSE]"
    if role == ROLE_LOCALITY:
        stats.add("LOCALITY")
        return "[LOCALITY]"
    if role == ROLE_LANDMARK:
        stats.add("LANDMARK")
        return "[LANDMARK]"
    if role == ROLE_VILLAGE:
        stats.add("VILLAGE")
        return "[VILLAGE]"
    if role == ROLE_TEHSIL:
        stats.add("TEHSIL")
        return "[TEHSIL]"
    if role == ROLE_DISTRICT:
        if plan and not plan.keep_district:
            stats.generalized += 1
            return "[DISTRICT]"
        return value
    if role == ROLE_PIN:
        stats.add("PIN")
        digits = norm_val.translate(_DEVA_DIGITS)
        if plan and not plan.keep_pin_prefix:
            return "[PIN]"
        if len(digits) == 6 and digits.isdigit():
            stats.generalized += 1
            return f"{digits[:3]}***"
        return "[PIN]"
    if role == ROLE_DOB:
        stats.add("DOB")
        if plan is not None and plan.generalization_level != "standard":
            return "[DOB]"
        parsed = parse_date_string(norm_val, preferred_order=date_order)
        if parsed and plan is not None:
            shifted = parsed + __import__("datetime").timedelta(days=plan.date_shift_days)
            stats.generalized += 1
            return shifted.isoformat() if date_order == "iso" else (
                shifted.strftime("%d/%m/%Y") if date_order == "dmy" else shifted.strftime("%m/%d/%Y")
            )
        return "[DOB]"
    if role in (ROLE_DATE, ROLE_ADMISSION_DATE, ROLE_DISCHARGE_DATE):
        if plan is not None:
            return shift_dates_in_text(
                value,
                plan.date_shift_days,
                hospital_hint=hospital_hint,
                is_suppressed=is_suppressed,
                preferred_order=date_order if date_order in ("dmy", "mdy") else None,
            )
        return value
    if role == ROLE_AGE:
        if norm_val.lstrip().startswith("-"):
            return value
        digits = re.sub(r"[^\d]", "", norm_val.translate(_DEVA_DIGITS))
        if digits and plan is not None:
            stats.generalized += 1
            if plan.age_band:
                return plan.age_band
            if is_suppressed:
                return "[AGE]"
            return format_age_band(int(digits))
        if digits:
            stats.generalized += 1
            return format_age_band(int(digits))
        return value
    if role == ROLE_AGE_SEX:
        m = re.match(r"^\s*(\d{1,3})\s*/\s*([MFO])\s*$", norm_val.translate(_DEVA_DIGITS), re.I)
        if m:
            age_v = int(m.group(1))
            band = (plan.age_band if (plan and plan.age_band) else ("[AGE]" if is_suppressed else format_age_band(age_v)))
            stats.generalized += 1
            g_str = "[GENDER]" if (plan is not None and not getattr(plan, "keep_gender", True)) else m.group(2).upper()
            return f"{band}/{g_str}"
        return clean_free_text(value, stats, plan=plan, hospital_hint=hospital_hint)
    if role == ROLE_GENDER:
        if plan is not None and not getattr(plan, "keep_gender", True):
            stats.generalized += 1
            return "[GENDER]"
        return value
    if role == ROLE_METADATA:
        # Redact source_path / clerk usernames carrying patient identifiers
        return clean_free_text(value, stats, plan=None, hospital_hint=hospital_hint) if (
            "/" in value or "\\" in value or "_" in value
        ) else value
    if role == ROLE_PROTECTED_CLINICAL:
        # Preserve clinical probes; still strip accidental direct IDs if any
        return clean_free_text(value, stats, plan=None, clean_only_direct_ids=True, hospital_hint=hospital_hint)
    if role == ROLE_ADDRESS:
        # Full composite address string: redact house/locality/landmark/village/relative/PIN, keep district if allowed
        cleaned = clean_free_text(value, stats, plan=plan, hospital_hint=hospital_hint)
        if plan and not plan.keep_district:
            cleaned = _SUPPRESS_DIST_EN_RX.sub("[DISTRICT]", cleaned)
            cleaned = _SUPPRESS_DIST_HI_RX.sub("[DISTRICT]", cleaned)
        return cleaned

    return clean_free_text(value, stats, plan=plan, hospital_hint=hospital_hint)
