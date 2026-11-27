"""Multilingual & context-aware PII detector with meaning-trap shields."""
from __future__ import annotations

import base64
import binascii
import html
import re
import urllib.parse
from dataclasses import dataclass, field

from rakshak.contracts import PrivacyPlan
from rakshak.ner_crf import tag_unanchored_person_spans
from rakshak.translit import fold_homoglyphs, is_valid_verhoeff
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
    # Generic hospital / lab / facility names (without hardcoding dev facility names)
    re.compile(
        r"\b(?:[A-Z][a-zA-Z.'-]{1,25}[ \t]+){0,4}"
        r"(?:Multispeciality[ \t]+Hospital|Charitable[ \t]+Hospital|District[ \t]+Hospital|Medical[ \t]+College|"
        r"Health[ \t]+Cent(?:er|re)|In-house[ \t]+Lab|Hospital|Pathology|Diagnostics|Labs?|Clinic|Sadan)"
        r"(?:[ \t]+(?:[A-Z][a-zA-Z.'-]{1,25}|Lab|Centre|Center)){0,2}"
        r"(?:[ \t]*\([^)\n]{1,40}\))?"
    ),
    # Medications, doses, frequencies, vitals, labs, ranges
    re.compile(r"(?i)\b(?:Tab\.?|Cap\.?|Inj\.?|Syp\.?|Drops|Oint\.?)[ \t]+[^\n;]+"),
    re.compile(r"\b[A-Z][a-zA-Z-]{2,}[ \t]+\d{1,4}[ \t]+(?:OD|BD|TDS|QID|HS|SOS|STAT)\b(?:[ \t]*x[ \t]*\d+[ \t]*(?:days?|wks?|weeks?|months?))?"),
    re.compile(r"\b[0-2]-[0-2]-[0-2]\b(?:[ \t]*x[ \t]*\d+[ \t]*(?:days?|wks?|weeks?|months?))?"),
    re.compile(r"(?i)\b\d+(?:\.\d+)?[ \t]*(?:mg/dL|g/dL|mg|mcg|ug|gm|g|mL|ml|IU|/uL|mmHg|U/L|mIU/L|°F)\b"),
    re.compile(r"(?i)\b(?:BP[ \t]*\d{2,3}/\d{2,3}|Pulse[ \t]*\d{2,3}(?:/min)?|SpO2[ \t]*\d{2,3}%|Temp[ \t]*\d{2,3}(?:\.\d+)?[ \t]*°?F)\b"),
    re.compile(
        r"(?<![A-Za-z0-9./+\-])(?![6-9]\d{4}-\d{5})(?!91-[6-9]\d{9})(?!\d{4}-\d{4}-\d{4})"
        r"\d{1,6}(?:\.\d+)?-\d{1,6}(?:\.\d+)?(?![A-Za-z0-9.\-])"
    ),
    re.compile(r"(?i)\b(?:x[ \t]*\d+[ \t]*(?:days?|wks?|weeks?|months?|yrs?)|for[ \t]+\d+[ \t]+(?:days?|wks?|weeks?|months?|yrs?))\b"),
    # ICD-10 (including U07.1 COVID-19) and generic PMJAY package codes
    re.compile(r"\b[A-Z]\d{2}(?:\.\d{1,2})?\b"),
    re.compile(r"\b[A-Z]{2}\d{3}[A-Z]\b"),
]

# --------------------------------------------------------------------------- Stage 2: Direct IDs
_DIRECT_ID_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ABHA_ADDRESS", re.compile(r"[\w.+-]+@(?:abdm|sbx|ndhm|phr|nha)\b", re.I)),
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    (
        "UPI_ID",
        re.compile(
            r"\b[\w.+-]{2,}@(?:ok(?:axis|icici|hdfcbank|axisbank|sbi)|"
            r"icici|hdfc\w*|sbi|axis\w*|kotak|ybl|ibl|axl|paytm|upi|apl|yapl|rapl|"
            r"pingpay|jio|fampay|aubank|idfc\w*|indus|yes\w*|federal|airtel|freecharge|"
            r"amazonpay|hsbc|barodampay|pnb|bob|boi|canara|union|wa(?:icici|axis|hdfc|sbi))\b",
            re.I,
        ),
    ),
    (
        "ABHA_NUMBER",
        re.compile(
            r"(?<!\d\.)(?<!\d)(?:(?!00)\d{2}[-\s.]\d{4}[-\s.]\d{4}[-\s.]\d{4}|"
            r"(?!00)(?!20\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01]))\d{14})(?!\d)"
        ),
    ),
    ("AADHAAR_MASKED", re.compile(r"(?i)(?<![A-Za-z0-9])(?:\([X*]{4}\)|[X*]{4})[\s.-]?[X*]{4}[\s.-]?\d{4}(?!\d)")),
    ("AADHAAR", re.compile(r"(?<!\d\.)(?<!\d)(?:\(\d{4}\)|\d{4})[\s.-]\d{4}[\s.-]\d{4}(?!\d)")),
    ("AADHAAR_LBL", re.compile(r"(?i)(?:\b(?:Aadhaar|Aadhar|UID|UIDAI|आधार)(?:[ \t]*(?:No\.?|Num(?:ber)?|ID))?[ \t]*[:=\-#][ \t]*)(\d{12})(?!\d)")),
    (
        "PHONE",
        re.compile(
            r"(?<!\d\.)(?<!\d)(?:(?:\(\+?91\)|\+?91|0091|0)[\s.-]?)?[6-9]\d{2}(?:[\s.-]?\d{3}[\s.-]?\d{4}|[\s.-]?\d{2}[\s.-]?\d{5}|\d{7})(?!\d)"
        ),
    ),
    ("PHONE", re.compile(r"(?<!\d\.)(?<!\d)0\d{2,4}[-.\s]\d{6,8}(?!\d)")),
    ("PAN", re.compile(r"(?i)\b[A-Z]{5}[ \t]?\d{4}[ \t]?[A-Z]\b")),
    (
        "INSURANCE_ID",
        re.compile(
            r"\b(?:(?:PMJAY|CGHS|ECHS|RSBY|TPA|POL|CLAIM|INS)[/_-][A-Z0-9]{2,}(?:[/_-]\d+)*|POL/\d+/\d+)\b",
            re.I,
        ),
    ),
    (
        "MRN",
        re.compile(
            r"(?i)\b(?!(?:OPD|POL|PMJAY|CGHS|ECHS|COVID|SARS|HIV|HBV|HCV|CD|IL|VIT|ICD|LOINC|SNOMED|TAB|CAP|INJ|SYP|"
            r"JAN|JANUARY|FEB|FEBRUARY|MAR|MARCH|APR|APRIL|MAY|JUN|JUNE|JUL|JULY|AUG|AUGUST|SEP|SEPT|SEPTEMBER|OCT|OCTOBER|NOV|NOVEMBER|DEC|DECEMBER)\b)"
            r"(?:[A-Z]{2,6}(?:\d{2,4})?[/_-]\d{2,4}[/_-]\d{3,8}|[A-Z]{1,6}(?:\d{2,4})?[-_/#]\d{4,10}|"
            r"20\d{2}[/_-][A-Z]{2,6}[/_-]\d{3,8}|(?:ADM|UHID|MRN|PID|REG)\d{4,9})\b"
        ),
    ),
]

# Compositional labeled ID fallback for unseen hospital/policy/chart/ref prefixes (group 2 = ID value, must contain a digit)
_LABELED_ID_RX = re.compile(
    r"(?i)((?:(?:UHID|MRN|IPD|OPD|PID|UID|ADM|NHS|ABDM|ABHA|Health|Patient|Pt\.?|Reg(?:n|\.?)|Registration|"
    r"Case|Chart|File|Episode|Bill|Claim|Policy|Member|Subscriber|Beneficiary|Card|Token|Ticket|Account|Ref(?:erence)?)"
    r"(?:[ \t]+(?:No\.?|Num(?:ber)?|ID|Code|Ref|#))?|पंजीकरण[ \t]+सं\.?|आईपीडी[ \t]+सं\.?|पॉलिसी[ \t]+सं\.?|मरी[ज़ज][ \t]+सं\.?)"
    r"(?:[ \t]*:[ \t]*|[ \t]*[=#|][ \t]*|[ \t]+-[ \t]+))"
    r"(?=[A-Za-z0-9/_#-]*\d|[ \t0-9]+\d)([A-Za-z0-9/_#-]{4,26}|\d{3,4}[ \t]\d{3,4}[ \t]\d{3,4})"
)

# --------------------------------------------------------------------------- Stage 3: Context-Anchored Names & Addresses
_HONORIFICS = (
    r"(?:ji|sahab|saheb|didi|bhaiya|behenji|bhai|amma|babu|chachu|chacha|chachi|"
    r"mama|mami|mausi|mausa|dadi|dada|nana|nani|kaka|kaki|bhabhi|papa|जी|साहब|दीदी|भैया)"
)

# Tuple: (entity_kind, regex, capture_group_index_to_redact)
_LABELED_NAME_PATTERNS: list[tuple[str, re.Pattern[str], int]] = [
    # Compositional Patient / Subject labels (Latin) - supports :, =, |, and " - " delimiters on explicit Name labels
    (
        "PATIENT_NAME",
        re.compile(
            r"(?im)(?<!Panel[ \t])(?<!Test[ \t])(?<!Drug[ \t])(?<!Lab[ \t])(?<!Hospital[ \t])(?<!Facility[ \t])(?<!Doctor[ \t])(?<!Department[ \t])(?<!Consultant[ \t])(?<!Physician[ \t])"
            r"((?:(?:Patient|Pt\.?|Subject|Client|Applicant|Account|Subscriber|Beneficiary|Member|Claimant|Insured|Person|Payee|Issued[ \t]+to|Examined[ \t]+for)"
            r"(?:[ \t]+(?:Full|Legal|First|Last|Display|Registered|Primary))?[ \t]+Name|(?:Full|Legal|Display)[ \t]+Name|Name)"
            r"(?:[ \t]*:[ \t]*|[ \t]*[=|][ \t]*|[ \t]+-[ \t]+)|"
            r"(?:Patient|Pt\.?|Subject|Client|Subscriber|Beneficiary|Member|Claimant|Payee|Issued[ \t]+to|Examined[ \t]+for)[ \t]*[:=][ \t]*)"
            r"([^\n,;|()]+?)(?=\s+(?:Age/Sex|Age|Sex|DOB|Birth|Date|Phone|Mob|Mobile|Tel|Aadhaar|ABHA|Gender|Address|UHID|IPD|MRN|Reg)\b|\s*(?:Age/Sex|Age|DOB|UHID|IPD|MRN)\s*[:=\-]|\n|$)"
        ),
        2,
    ),
    # Compositional Patient labels (Devanagari)
    (
        "PATIENT_NAME",
        re.compile(
            r"(?m)((?:मरी[ज़ज][ \t]*(?:का[ \t]+नाम)?|रोगी[ \t]*(?:का[ \t]+नाम)?|अभ्यर्थी[ \t]*(?:का[ \t]+नाम)?|नाम)"
            r"(?:[ \t]*:[ \t]*|[ \t]*[=|][ \t]*|[ \t]+-[ \t]+))"
            r"([^\n,;|()]+?)(?=\s{2,}|\s+(?:आयु|उम्र|लिंग|पता|मोबाइल|जन्म)\b|\s*(?:आयु|उम्र|लिंग|पता|मोबाइल)\s*[:=\-]|\n|$)"
        ),
        2,
    ),
    # Compositional Relative / Kin / Caregiver labels (Devanagari & English)
    (
        "RELATIVE_NAME",
        re.compile(
            r"(?im)((?:पिता[ \t]+का[ \t]+नाम|पति[ \t]+का[ \t]+नाम|अभिभावक(?:[ \t]+का[ \t]+नाम)?|"
            r"(?:Father|Mother|Husband|Wife|Spouse|Guardian|Attendant|Caregiver|Emergency[ \t]+Contact|Nominee|Next[ \t]+of[ \t]+Kin|Kin|Relative|Witness)"
            r"(?:(?:'s)?[ \t]+(?:Full[ \t]+)?Name)?)(?:[ \t]*:[ \t]*|[ \t]*[=|][ \t]*|[ \t]+-[ \t]+))([^\n,;|()]+)"
        ),
        2,
    ),
    # Relation markers (S/o, D/o, W/o, C/O <person>) - support mixed/lowercase too
    (
        "RELATIVE_NAME",
        re.compile(
            r"(?m)(\b(?:S/o|D/o|W/o|C/o|S/O|D/O|W/O|C/O)[ \t]+)"
            r"([A-Za-z][a-zA-Z.'-]*(?:[ \t]+[A-Za-z][a-zA-Z.'-]*){0,3}|[\u0900-\u097F]+(?:[ \t]+[\u0900-\u097F]+){0,2})"
        ),
        2,
    ),
    # Attendant / Guardian / Nominee explicit label
    (
        "RELATIVE_NAME",
        re.compile(
            r"(?im)((?:Attendant|Guardian|Nominee|Next[ \t]+of[ \t]+kin)[ \t]*:[ \t]*)"
            r"([A-Za-z][a-zA-Z.'-]+(?:[ \t]+[A-Za-z][a-zA-Z.'-]+){0,3}|[\u0900-\u097F]+(?:[ \t]+[\u0900-\u097F]+){0,2})"
        ),
        2,
    ),
    # Relation prefix without colon ("Wife Priya Chaturvedi", "Father Debashis Mahapatra", "परिजन Pema Bhutia")
    (
        "RELATIVE_NAME",
        re.compile(
            r"(?m)(?:^|(?<=[\s(\[.,;]))((?:Wife|Husband|Father|Mother|Spouse|Attendant|Guardian|Nominee)[ \t]+)"
            r"(?!(?:of|to|for|from|in|on|at|with|was|had|is|are|reported|died|verified|signed)\b)"
            r"([A-Z][a-zA-Z.'-]+(?:[ \t]+(?!(?:and|or|on|at|in|for|to|of|from|phone|mob|mobile|District|PIN)\b)[A-Z][a-zA-Z.'-]+){0,3})"
        ),
        2,
    ),
    (
        "RELATIVE_NAME",
        re.compile(
            r"(?m)((?:परिजन|पत्नी|पति|पिता|माता|अभिभावक)[ \t]+)"
            r"([A-Z][a-zA-Z.'-]+(?:[ \t]+(?!(?:and|or|on|at|in|for|to|of|from|phone)\b)[A-Z][a-zA-Z.'-]+){0,3}|[\u0900-\u097F]{2,}(?:[ \t]+[\u0900-\u097F]{2,}){0,2})"
            r"(?=[ \t]+(?:ने|को|का|की|के|से)\b|[ \t]*[,।\n]|$)"
        ),
        2,
    ),
    # Patient opening "<Full Name>, <age> yrs" in clinical summaries
    (
        "PATIENT_NAME",
        re.compile(
            r"(?m)^([A-Z][a-zA-Z.'-]+(?:[ \t]+[A-Z][a-zA-Z.'-]+){1,3})"
            r"(?=[ \t]*,[ \t]*\d{1,3}[ \t]*(?:yrs?|years?|y/[MFO]))"
        ),
        1,
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
    # Hinglish honorific phrases ("Shahid sahab", "Asha ji", "ramesh bhai") avoiding prepositions ("of Jaspreet babu")
    (
        "PATIENT_NAME",
        re.compile(
            rf"\b(?!(?:of|for|by|to|from|in|on|at|the|and|with|report|patient|doctor|dr)\b)"
            rf"([A-Za-z][a-zA-Z.'-]{{1,}}(?:[ \t]+(?!(?:of|for|by|to|from|in|on|at|the|and|with)\b)[A-Za-z][a-zA-Z.'-]{{1,}})?[ \t]+(?i:{_HONORIFICS}))\b",
            re.I,
        ),
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
    # Hindi narrative sentence subject "<Name> को <duration/symptom>"
    (
        "PATIENT_NAME",
        re.compile(r"([\u0900-\u097F]{2,}(?:[ \t]+[\u0900-\u097F]{2,}){1,2})(?=[ \t]+को[ \t]+(?:\d+[ \t]*(?:दिन|हफ्ते|सप्ताह|माह|साल)|तेज|कल|बुखार|दर्द))"),
        1,
    ),
]

_ADDRESS_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("HOUSE", re.compile(r"(?im)\b(?:H\.?[ \t]*No\.?|House(?:[ \t]+No\.?)?|Makan[ \t]+No\.?)[: \t]*[A-Za-z0-9/-]+")),
    ("HOUSE", re.compile(r"(?m)मकान[ \t]+नं\.?[: \t]*\d+[A-Za-z0-9/-]*")),
    ("LOCALITY", re.compile(r"(?im)\bWard[ \t]+\d+\b(?![ \t]*\()")),
    ("LOCALITY", re.compile(r"(?im)\bMohalla[: \t]+[^,\n-]+?(?=,|[ \t]+(?:near|opp|behind|Vill|Gram|Tehsil|Distt|District|PIN)|$)")),
    ("LOCALITY", re.compile(r"(?m)मोहल्ला[: \t]+[^,\n-]+?(?=,|[ \t]+(?:तहसील|जिला|जनपद|ग्राम|के[ \t]+पास|के[ \t]+पीछे|के[ \t]+सामने)|$)")),
    ("LANDMARK", re.compile(r"(?im)\b(?:near|opp\.?|opposite|behind)[ \t]+[^,\n-]+?(?=,|[ \t]+(?:Vill|Gram|Tehsil|Distt|District|PIN)|$)")),
    ("LANDMARK", re.compile(r"(?m)[^\s,।]+(?:[ \t]+[^\s,।]+){0,3}[ \t]+के[ \t]+(?:पास|पीछे|सामने)")),
    ("VILLAGE", re.compile(r"(?im)\b(?:Vill\.?|Village|Gram)[: \t]+[^,\n-]+?(?=,|[ \t]+(?:Tehsil|Block|Mandal|Taluka|Distt|District|PIN)|$)")),
    ("VILLAGE", re.compile(r"(?m)ग्राम[: \t]+[^,\n-]+?(?=,|[ \t]+(?:तहसील|ब्लॉक|जिला|जनपद|पिन)|$)")),
    ("TEHSIL", re.compile(r"(?im)\b(?:Tehsil|Taluka|Block|Mandal)[: \t]+[^,\n-]+?(?=,|[ \t]+(?:Distt|District|PIN)|$)")),
    ("TEHSIL", re.compile(r"(?m)(?:तहसील|ब्लॉक)[: \t]+[^,\n-]+?(?=,|[ \t]+(?:जिला|जनपद|पिन)|$)")),
    # Street / lane / road / marg / gali / sector addresses (guard clinical/hospital nouns generically)
    (
        "LOCALITY",
        re.compile(
            r"(?m)\b(?:(?i:Lane|Street|St\.?|Sector|Phase|Block|Gali)[ \t]+[A-Za-z0-9][A-Za-z0-9 .'-]{1,30}|"
            r"(?!(?:Hospital|Medical|Clinic|College|District|Labs?|Pathology|Diagnostics|Health|Trust|Centre|Center)\b)"
            r"[A-Z][A-Za-z0-9.'-]*(?:[ \t]+[A-Z][A-Za-z0-9.'-]*){0,2}[ \t]+(?:Road|Rd\.?|Marg|Lane|Street|St\.?|Gali|Chowk|Nagar|Vihar|Enclave|Colony))\b"
        ),
    ),
]

_PIN_IN_ADDRESS_RX = re.compile(
    r"(?i)(?:-\s|(?:PIN(?:\s*Code)?|Postal\s*Code|Zip|पिन(?:\s*कोड)?)[:\s-]\s*|(?:Distt?\.?|District|ghar|vill\.?|village|gram|tehsil|taluka|road|marg|nagar)\s+[A-Za-z\s,]{2,30}|(?:जिला|जनपद)\s*[:.]?\s*[\u0900-\u097F\s,]{2,25})(\b[1-8]\d{5}\b)"
)
_STANDALONE_PIN_RX = re.compile(r"(?:^|[-\s,])([1-8]\d{5})(?=\s*$|\s*[.,\n])", re.M)

_FREE_AGE_EN_RX = re.compile(
    r"(?i)(\b(?:Age/Sex|Age|Umar|Umra|Vayas)\s*[:=\-]\s*)(\d{1,3})(\s*(?:Yrs?|Years?|Y)?(?:\s*/\s*(?:Male|Female|[MFO]))?)\b"
)
_FREE_AGE_WA_RX = re.compile(r"(\b)(\d{1,3})(\s*Y(?:rs?)?\s*/\s*[MFO]\b)", re.I)
_FREE_AGE_HI_RX = re.compile(r"((?:आयु|उम्र)\s*[:=\-]\s*)(\d{1,3})(\s*(?:वर्ष|साल)(?:\s*/\s*(?:पुरुष|पुरूष|महिला|स्त्री|नर|मादा|अन्य))?)")


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
_SUPPRESS_DIST_EN_RX = re.compile(r"(?i)\b(?:Distt?\.?|District)[ \t]*[:.]?[ \t]+[A-Za-z]+(?:[ \t]+[A-Za-z]+){0,2}(?=[ \t]*(?:,|-|\.(?:\s|$)|\n|$))")
_SUPPRESS_DIST_HI_RX = re.compile(
    r"(?:जिला|जनपद)[ \t]*[:.]?[ \t]+(?!(?:अस्पताल|चिकित्सालय|मेडिकल[ \t]+कॉलेज)\b)[\u0900-\u097F]+(?:[ \t]+[\u0900-\u097F]+){0,1}(?=[ \t]*(?:,|-|\.|।|\n|$))"
)
_FAST_SAFE_TOKEN_RX = re.compile(r"^[A-Za-z_-]{1,36}$")
_VERHOEFF_AADHAAR_RX = re.compile(r"(?<!\d\.)(?<!\d)([2-9]\d{11})(?!\d)")
_INLINE_B64_RX = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/=])")
_INLINE_HEX_RX = re.compile(r"\b(?:0x)?((?:[0-9a-fA-F]{2}){10,})\b")
_INLINE_ESC_HINT_RX = re.compile(r"(?:\\u[0-9a-fA-F]{4}|&#(?:\d+|x[0-9a-fA-F]+);|(?:%[0-9a-fA-F]{2}.*){2,})")
_INLINE_UESC_RX = re.compile(r"\\u([0-9a-fA-F]{4})")
_INLINE_HTML_RX = re.compile(r"&#(?:\d+|x[0-9a-fA-F]+);")
_INLINE_PCT_RX = re.compile(r"(?:%[0-9a-fA-F]{2})")


def _is_readable_payload(txt: str) -> bool:
    if not txt or len(txt) < 4:
        return False
    bad = sum(1 for ch in txt if ord(ch) < 0x20 and ch not in "\t\r\n")
    if bad > 0:
        return False
    letters = sum(1 for ch in txt if ch.isalnum() or ch.isspace() or ch in ",.;:!?@#/-_()[]'\"")
    return (letters / len(txt)) >= 0.85


def _scrub_inline_encoded_payloads(
    text: str,
    stats: DetectionStats,
    plan: PrivacyPlan | None,
    hospital_hint: str,
    preferred_order: str | None,
    allow_ml_fallback: bool,
    cluster_redact: tuple[tuple[str, str], ...],
    decode_depth: int,
) -> str:
    """Decode and sanitize inline Base64, hex, \\uXXXX, HTML-entity, and %-encoded segments."""
    if decode_depth >= 2 or not text:
        return text

    work = text
    # 1. Inline Base64 runs (re-encode cleaned payload to keep container intact)
    if len(work) >= 24 and _INLINE_B64_RX.search(work):
        def _b64_repl(m: re.Match[str]) -> str:
            tok = m.group(0)
            padded = tok + "=" * (-len(tok) % 4)
            try:
                raw = base64.b64decode(padded, validate=True)
                dec = raw.decode("utf-8", errors="strict")
            except (binascii.Error, UnicodeDecodeError, ValueError):
                return tok
            if not _is_readable_payload(dec):
                return tok
            cleaned = clean_free_text(
                dec, stats, plan=plan, hospital_hint=hospital_hint,
                preferred_order=preferred_order, allow_ml_fallback=allow_ml_fallback,
                cluster_redact=cluster_redact, _decode_depth=decode_depth + 1,
            )
            if cleaned == dec:
                return tok
            return base64.b64encode(cleaned.encode("utf-8")).decode("ascii")

        work = _INLINE_B64_RX.sub(_b64_repl, work)

    # 2. Inline Hex strings (>= 10 bytes) encoding printable UTF-8 PII
    if _INLINE_HEX_RX.search(work):
        def _hex_repl(m: re.Match[str]) -> str:
            full = m.group(0)
            hex_body = m.group(1)
            try:
                dec = bytes.fromhex(hex_body).decode("utf-8", errors="strict")
            except (ValueError, UnicodeDecodeError):
                return full
            if not _is_readable_payload(dec):
                return full
            cleaned = clean_free_text(
                dec, stats, plan=plan, hospital_hint=hospital_hint,
                preferred_order=preferred_order, allow_ml_fallback=allow_ml_fallback,
                cluster_redact=cluster_redact, _decode_depth=decode_depth + 1,
            )
            if cleaned == dec:
                return full
            pref = "0x" if full.startswith("0x") else ""
            return pref + cleaned.encode("utf-8").hex()

        work = _INLINE_HEX_RX.sub(_hex_repl, work)

    # 3. Line-level \uXXXX / HTML numeric entities / %-encoding
    if _INLINE_ESC_HINT_RX.search(work):
        out_lines: list[str] = []
        for line in work.splitlines(keepends=True):
            if "\x00SHIELD_" in line or not _INLINE_ESC_HINT_RX.search(line):
                out_lines.append(line)
                continue
            dec_line = line
            if _INLINE_UESC_RX.search(dec_line):
                dec_line = _INLINE_UESC_RX.sub(lambda m: chr(int(m.group(1), 16)), dec_line)
            if _INLINE_HTML_RX.search(dec_line):
                dec_line = html.unescape(dec_line)
            if len(_INLINE_PCT_RX.findall(dec_line)) >= 2:
                dec_line = urllib.parse.unquote(dec_line)
            if dec_line != line and _is_readable_payload(dec_line.strip()):
                cleaned = clean_free_text(
                    dec_line, stats, plan=plan, hospital_hint=hospital_hint,
                    preferred_order=preferred_order, allow_ml_fallback=allow_ml_fallback,
                    cluster_redact=cluster_redact, _decode_depth=decode_depth + 1,
                )
                if cleaned != dec_line:
                    out_lines.append(cleaned)
                    continue
            out_lines.append(line)
        work = "".join(out_lines)

    return work


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
        rep = replacement_fn(m, raw_sub) if replacement_fn else f"[{kind}]"
        out.append(text[pos:s])
        out.append(rep)
        pos = e
        if rep != raw_sub:
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


def _apply_crf_fallback(text: str, stats: DetectionStats) -> str:
    """Run guarded python-crfsuite fallback on unanchored free-text prose only."""
    spans = tag_unanchored_person_spans(text)
    if not spans:
        return text
    out: list[str] = []
    pos = 0
    for s, e, kind in spans:
        if s < pos or s >= e:
            continue
        raw = text[s:e]
        if "\x00SHIELD_" in raw or "[" in raw or "]" in raw:
            continue
        out.append(text[pos:s])
        out.append(f"[{kind}]")
        pos = e
        stats.add(kind)
    out.append(text[pos:])
    return "".join(out)


def _apply_cluster_redact(
    text: str,
    cluster_redact: tuple[tuple[str, str], ...],
    stats: DetectionStats,
) -> str:
    """Scrub already-linked cluster names, relatives, and suppressed QIs after shielding."""
    if not cluster_redact or not text:
        return text
    out = text
    for kind, surf in sorted(cluster_redact, key=lambda kv: (-len(kv[1]), kv[0], kv[1])):
        if not surf or len(surf.strip()) < 3:
            continue
        esc = re.escape(surf.strip())
        if surf.isascii():
            rx = re.compile(rf"(?i)(?<![A-Za-z0-9_]){esc}(?![A-Za-z0-9_])")
        else:
            rx = re.compile(rf"(?:(?<=^)|(?<=[\s,.;:!?()\[\]\"'/-])){esc}(?=$|[\s,.;:!?()\[\]\"'/।-])")
        out = _apply_shadow_sub(out, rx, kind, stats)
    return out


def clean_free_text(
    text: str,
    stats: DetectionStats,
    plan: PrivacyPlan | None = None,
    clean_only_direct_ids: bool = False,
    hospital_hint: str = "",
    preferred_order: str | None = None,
    allow_ml_fallback: bool = True,
    cluster_redact: tuple[tuple[str, str], ...] = (),
    _decode_depth: int = 0,
) -> str:
    """Detect and redact PII in free text while protecting clinical meaning traps."""
    if not text or _FAST_SAFE_TOKEN_RX.fullmatch(text):
        return text

    # Fold zero-width chars, Cyrillic/Greek homoglyphs, and Latin diacritics while keeping Devanagari digits intact
    work_in = text if text.isascii() else fold_homoglyphs(text, keep_deva_digits=True)

    if clean_only_direct_ids:
        res, direct_shields = _shield_text(work_in)
        for kind, rx in _DIRECT_ID_PATTERNS:
            if kind == "MRN":
                continue
            if kind == "AADHAAR_LBL":
                res = _apply_shadow_sub(res, rx, "AADHAAR", stats, group_idx=1)
                continue
            if kind == "PHONE":
                res = _apply_shadow_sub(
                    res, _VERHOEFF_AADHAAR_RX, "AADHAAR", stats,
                    replacement_fn=lambda m, raw: "[AADHAAR]" if is_valid_verhoeff(m.group(0)) else raw,
                )
            res = _apply_shadow_sub(res, rx, kind, stats)
        return _unshield_text(res, direct_shields)

    # 1. Shield protected clinical expressions first
    work, shields = _shield_text(work_in)

    # 1.5 Inline encoded payloads (base64 / hex / \uXXXX / HTML entities / %-encoding)
    work = _scrub_inline_encoded_payloads(
        work, stats, plan, hospital_hint, preferred_order, allow_ml_fallback, cluster_redact, _decode_depth
    )

    # 2. Direct structured IDs (shadow-matched across ASCII & Devanagari digits; Verhoeff bare Aadhaar before PHONE)
    verhoeff_done = False
    for kind, rx in _DIRECT_ID_PATTERNS:
        if kind == "AADHAAR_LBL":
            work = _apply_shadow_sub(work, rx, "AADHAAR", stats, group_idx=1)
            continue
        if kind == "PHONE" and not verhoeff_done:
            verhoeff_done = True
            work = _apply_shadow_sub(
                work, _VERHOEFF_AADHAAR_RX, "AADHAAR", stats,
                replacement_fn=lambda m, raw: "[AADHAAR]" if is_valid_verhoeff(m.group(0)) else raw,
            )
        work = _apply_shadow_sub(work, rx, kind, stats)

    # Labeled unseen-format MRNs
    work = _apply_shadow_sub(work, _LABELED_ID_RX, "MRN", stats, group_idx=2)

    # 3. Context-anchored patient & relative names
    _non_name_label_vals = frozenset({
        "unknown", "none", "na", "n/a", "nil", "null", "normal", "stable",
        "positive", "negative", "pending", "approved", "settled", "routine",
    })
    for kind, rx, grp in _LABELED_NAME_PATTERNS:
        def _name_rep(m: re.Match[str], raw_sub: str, k: str = kind) -> str:
            low_sub = raw_sub.strip().casefold()
            if (raw_sub.strip().startswith("[") and raw_sub.strip().endswith("]")) or low_sub in _non_name_label_vals:
                return raw_sub
            return f"[{k}]"
        work = _apply_shadow_sub(work, rx, kind, stats, replacement_fn=_name_rep, group_idx=grp)

    # 4. Address components (House, Locality, Landmark, Village, Tehsil, PIN) before CRF fallback
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

    # 4.5 Guarded python-crfsuite fallback for unanchored multilingual prose names/relatives/locations/IDs
    if allow_ml_fallback:
        work = _apply_crf_fallback(work, stats)

    # 4.6 Exact cluster-known names, relatives, and suppressed QIs (shield-protected above)
    if cluster_redact:
        work = _apply_cluster_redact(work, cluster_redact, stats)

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
        work = _shift_month_name_dates(
            work,
            plan.date_shift_days,
            is_suppressed=(plan.generalization_level != "standard"),
        )
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
        # Skip dotted version tokens (1.2.15) unless explicitly on a dated/DOB line
        line_start = shadow.rfind("\n", 0, m.start()) + 1
        line_prefix = shadow[line_start:m.start()].lower()
        is_dated_ctx = any(k in line_prefix for k in ("dob", "birth", "born", "bday", "date", "admis", "admit", "disch", "visit", "reg", "collected", "reported", "जन्म", "तारीख", "दिनांक", "भर्ती", "छुट्टी"))
        if len(parts[0]) != 4 and len(parts[2]) == 2 and int(parts[2]) < 20 and "." in tok and not is_dated_ctx:
            return tok
        parsed = parse_date_string(tok, preferred_order=preferred)
        if parsed is None or not (1900 <= parsed.year <= 2035):
            return tok
        # Check line context for DOB: truncate to YYYY on standard tier (never shift with clinical dates)
        if any(k in line_prefix for k in ("dob", "birth", "born", "bday", "janm", "जन्म")):
            return "[DOB]" if is_suppressed else f"{parsed.year:04d}"
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


def _shift_month_name_dates(text: str, shift_days: int, is_suppressed: bool = False) -> str:
    """Shift textual dates in a single left-to-right pass (zero double-shifting), redacting DOB if suppressed."""
    from rakshak.privacy import _TEXT_MONTH_DATE_RE, _US_TEXT_MONTH_DATE_RE, _YMD_TEXT_MONTH_DATE_RE, shift_date_token

    out: list[str] = []
    pos = 0
    n = len(text)
    rxs = (_YMD_TEXT_MONTH_DATE_RE, _TEXT_MONTH_DATE_RE, _US_TEXT_MONTH_DATE_RE)
    while pos < n:
        best_m: re.Match[str] | None = None
        for rx in rxs:
            m = rx.search(text, pos)
            if m is not None and (
                best_m is None or m.start() < best_m.start() or (m.start() == best_m.start() and m.end() > best_m.end())
            ):
                best_m = m
        if best_m is None:
            out.append(text[pos:])
            break
        out.append(text[pos:best_m.start()])
        raw_tok = best_m.group(0)
        ls = text.rfind("\n", 0, best_m.start()) + 1
        seg = re.split(r"[|;]", text[ls:best_m.start()])[-1].lower()
        if any(k in seg for k in ("dob", "birth", "born", "bday", "janm", "जन्म")):
            if is_suppressed:
                out.append("[DOB]")
            else:
                m_yr = re.search(r"\b((?:19|20)\d{2})\b", raw_tok)
                out.append(m_yr.group(1) if m_yr else "[DOB]")
            pos = best_m.end()
            continue
        out.append(shift_date_token(raw_tok, shift_days))
        pos = best_m.end()
    return "".join(out)


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
    cluster_redact: tuple[tuple[str, str], ...] = (),
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
        # Inject linked pseudonym into primary hospital/patient ID cells (alphanumeric or pure-digit MRNs)
        stats.add("MRN")
        if plan and plan.pseudonym and re.search(r"(?:[A-Za-z]{2,8}[/-]?\d{3,}|^\d{4,12}$)", norm_val):
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
        if parsed is not None:
            stats.generalized += 1
            return f"{parsed.year:04d}"
        m_yr = re.fullmatch(r"(\d{4})", norm_val.translate(_DEVA_DIGITS))
        if m_yr:
            stats.generalized += 1
            return m_yr.group(1)
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
        return clean_free_text(value, stats, plan=plan, hospital_hint=hospital_hint, cluster_redact=cluster_redact)
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

    return clean_free_text(value, stats, plan=plan, hospital_hint=hospital_hint, cluster_redact=cluster_redact)
