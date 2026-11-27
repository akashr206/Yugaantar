"""Lightweight offline python-crfsuite fallback tagger for unanchored prose PII.

Predicts multi-label BIO entities (PER, REL, LOC, ID) behind Stage-1 clinical
shields and strict post-inference medical/schema precision guards so clinical
probes, doctors, hospitals, and eponyms are never altered.
"""
from __future__ import annotations

import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import pycrfsuite
except ImportError:  # pragma: no cover - optional runtime fallback if dependency missing
    pycrfsuite = None  # type: ignore[assignment]

from rakshak.translit import fold_homoglyphs, phonetic_normalize_token

_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "pii_ner.crfsuite"
_TAGGER_LOCK = threading.Lock()
_TAGGER_CACHE: Any = None
_TAGGER_LOADED = False

# Tokenize shields, [TAGS], hyphen/apostrophe compound names, and Devanagari words atomically
_TOK_RX = re.compile(
    r"\x00SHIELD_\d+\x00|\[[A-Z_]{2,24}\]|[\u0900-\u097F]+(?:['’-][\u0900-\u097F]+)*|"
    r"[^\W_]+(?:['’-][^\W_]+)*|\d+(?:[.,:/-]\d+)*|[^\s]",
    re.UNICODE,
)
_DEVA_RX = re.compile(r"[\u0900-\u097F]")

# Soft chemical/medical suffix indicator passed to CRF (covers broader stems without hard-blocking names)
_CLINICAL_CHEM_SUFFIX_RX = re.compile(
    r"(?i)(?:mycin|cillin|pril|olol|statin|sartan|prazole|gliptin|formin|dipine|"
    r"zolam|pam|mab|nib|vir|zole|conazole|ase|emia|uria|osis|itis|pathy|gram|metry|algia|ectomy|plasty)$"
)
# Unambiguous pharmacological/pathology suffixes safe for hard post-guard (never rejects -vir/-gram/-pam names)
_STRICT_MED_SUFFIX_RX = re.compile(
    r"(?i)(?:mycin|cillin|pril|olol|statin|sartan|prazole|gliptin|formin|dipine|"
    r"zolam|conazole|emia|uria|osis|itis|pathy|algia|ectomy|plasty)$"
)

_SALUTATION_WORDS = frozenset({
    "mr", "mrs", "ms", "miss", "dr", "prof", "shri", "smt", "shrimati", "sushri",
    "master", "baby", "late", "श्री", "श्रीमती", "सुश्री", "डॉ",
})
_ROLE_PREFIX_STOPWORDS = frozenset({
    "policyholder", "passenger", "claimant", "subscriber", "beneficiary", "applicant",
    "member", "client", "customer", "holder", "account", "user", "admin", "clerk",
})
_REL_CUES = frozenset({
    "s/o", "d/o", "w/o", "c/o", "wife", "husband", "son", "daughter", "father",
    "mother", "spouse", "guardian", "attendant", "nominee", "caregiver",
    "पत्नी", "पति", "पुत्र", "पुत्री", "पिता", "माता", "परिजन", "अभिभावक",
})
_HINDI_CASE_WORDS = frozenset({
    "को", "का", "की", "के", "ने", "से", "में", "पर", "और", "है", "हैं", "था", "थी", "थे",
    "ko", "ka", "ki", "ke", "ne", "se", "aur", "hai", "hain", "tha", "thi", "the",
})
_PREPOSITION_START_WORDS = frozenset({
    "on", "in", "at", "by", "for", "from", "to", "with", "during", "after", "before",
    "since", "while", "when", "where", "under", "yesterday", "today", "tomorrow", "follow-up",
    "follow", "admitted", "discharged", "evaluated", "referred", "young", "elderly", "child",
    "pls", "please", "recheck", "review", "check", "verify", "update",
})
_PLACE_ORG_WORDS = frozenset({
    "hospital", "clinic", "polyclinic", "dispensary", "institute", "sciences", "medical",
    "centre", "center", "desk", "department", "dept", "unit", "ward", "icu", "opd", "ipd",
    "endocrine", "endocrinology", "cardiology", "pulmonology", "neurology", "pediatric",
    "pediatrics", "diagnostics", "laboratory", "labs", "pharmacy", "emergency", "summary",
    "progress", "note", "notes", "round", "rounds", "session", "impression",
    "system", "gateway", "billing", "registry", "authority", "commission", "directorate",
    "corridor", "council", "board", "bank", "portal", "automated", "southern", "northern",
    "eastern", "western", "central", "national", "state", "district", "gramin", "sahakari",
    "comprehensive", "letter", "switch", "module", "service", "platform", "agency", "trust",
    "foundation", "audit", "regulatory", "memo", "protocol", "guideline", "enterprise",
    "case", "status", "stage", "grade", "level", "type", "class", "mode", "plan", "goal",
    "panel", "profile", "section", "claim", "coverage", "detail", "details", "info",
    "record", "entry", "issue", "query", "item", "chart", "episode", "referral", "finding", "findings",
})
_CLINICAL_CUE_WORDS = frozenset({
    "tab", "cap", "inj", "syp", "ointment", "drops", "tablet", "capsule", "injection",
    "diagnosed", "diagnosis", "disease", "syndrome", "palsy", "lymphoma", "fever", "pain",
    "cough", "bp", "pulse", "spo2", "temp", "hemoglobin", "creatinine", "glucose", "mg",
    "ml", "gm", "mcg", "od", "bd", "tds", "qid", "ward", "bed", "hospital", "clinic",
    "presented", "reported", "complained", "showed", "completed", "stabilized", "brought",
    "evaluated", "prescribed", "advised", "slept", "ambulating", "responding", "verified",
    "बुखार", "दर्द", "खांसी", "दवा", "जांच", "अस्पताल", "चिकित्सालय", "वार्ड", "रोग", "निदान", "कमजोरी",
    "bukhar", "dard", "khansi", "dawa", "ilaj", "kamzori", "chakkar", "sujan",
})

# Non-person tokens that must NEVER be tagged as unanchored patient names
_NON_PERSON_STOPWORDS = frozenset({
    "patient", "patients", "doctor", "doctors", "dr", "nurse", "staff", "consultant",
    "attendant", "relative", "guardian", "nominee", "beneficiary", "subscriber",
    "member", "client", "customer", "applicant", "holder", "account", "user", "admin",
    "policyholder", "passenger", "claimant", "clerk",
    "hospital", "clinic", "polyclinic", "dispensary", "medical", "health", "centre", "center",
    "institute", "sciences", "pathology", "lab", "labs", "diagnostics", "department", "desk",
    "medicine", "surgery", "pediatrics", "pediatric", "cardiology", "neurology", "orthopedics",
    "gynecology", "endocrine", "endocrinology", "pulmonology", "unit", "ward", "bed", "icu",
    "opd", "ipd", "emergency", "progress", "impression", "rounds", "follow", "follow-up", "followup",
    "system", "gateway", "billing", "registry", "authority", "commission", "directorate",
    "corridor", "council", "board", "portal", "automated", "southern", "northern",
    "eastern", "western", "central", "national", "gramin", "sahakari", "comprehensive",
    "letter", "switch", "module", "service", "platform", "agency", "trust", "foundation",
    "audit", "regulatory", "memo", "protocol", "guideline", "enterprise",
    "case", "status", "stage", "grade", "level", "type", "class", "mode", "plan", "goal",
    "panel", "profile", "section", "claim", "coverage", "detail", "details", "info",
    "record", "entry", "issue", "query", "item", "chart", "episode", "referral", "finding",
    "findings", "active", "routine", "care", "dose", "rate", "visit", "checkup", "exam",
    "on", "in", "at", "by", "for", "from", "to", "of", "the", "and", "or", "with", "during",
    "after", "before", "since", "while", "when", "where", "under", "young", "child", "elderly",
    "pls", "please", "recheck", "check", "update", "verify", "she", "he", "they", "his", "her",
    "blood", "pressure", "heart", "rate", "pulse", "temperature", "oxygen", "saturation",
    "sugar", "fasting", "random", "serum", "plasma", "urea", "bilirubin", "creatinine",
    "hemoglobin", "haemoglobin", "platelet", "count", "test", "result", "results", "normal",
    "abnormal", "positive", "negative", "high", "low", "mild", "moderate", "severe",
    "acute", "chronic", "stable", "unstable", "improved", "discharged", "admitted",
    "referred", "reviewed", "advised", "review", "history", "chief",
    "complaint", "complaints", "fever", "pain", "cough", "cold", "vomiting", "headache",
    "weakness", "nausea", "diarrhea", "diarrhoea", "swelling", "rash", "dizziness",
    "diabetes", "hypertension", "tuberculosis", "typhoid", "dengue", "malaria", "rabies",
    "asthma", "pneumonia", "anemia", "anaemia", "jaundice", "sepsis", "covid", "disease",
    "syndrome", "palsy", "lymphoma", "thyroid", "hypothyroidism", "thyroiditis", "leptospirosis",
    "kidney", "liver", "lung", "lungs", "chest", "abdomen", "head", "brain", "spine",
    "oral", "daily", "twice", "thrice", "days", "day", "week", "weeks", "month", "months",
    "year", "years", "yrs", "date", "time", "age", "sex", "gender", "male", "female", "other",
    "unknown", "none", "address", "village", "district", "tehsil", "state", "india", "pin",
    "code", "phone", "mobile", "email", "aadhaar", "abha", "pan", "uhid", "mrn", "report",
    "summary", "note", "notes", "remarks", "advice", "prescription", "receipt", "invoice",
    "bill", "amount", "balance", "total", "salary", "payment", "bank", "branch", "transaction", "order",
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug",
    "sep", "sept", "oct", "nov", "dec", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "today", "yesterday", "tomorrow",
    "मरीज", "रोगी", "मरीज़", "डॉक्टर", "डॉ", "चिकित्सक", "नर्स", "अस्पताल", "चिकित्सालय",
    "वार्ड", "बिस्तर", "दवा", "दवाई", "गोली", "कैप्सूल", "इंजेक्शन", "सिरप", "जांच", "रिपोर्ट",
    "बुखार", "दर्द", "खांसी", "सर्दी", "उल्टी", "दस्त", "कमजोरी", "चक्कर", "सूजन", "निदान",
    "इतिहास", "सलाह", "छुट्टी", "भर्ती", "तारीख", "आयु", "उम्र", "लिंग", "पुरुष", "महिला", "पता",
    "ग्राम", "गाँव", "गांव", "तहसील", "जिला", "जनपद", "पिन", "मोबाइल", "फोन", "आधार",
    "पंजीकरण", "सामान्य", "सकारात्मक", "नकारात्मक", "कल", "आज", "सुबह", "शाम", "रात",
    "दिन", "सप्ताह", "हफ्ते", "महीना", "महीने", "साल", "वर्ष", "तेज", "हल्का", "गंभीर",
    "सीधे", "वाक्य", "पिछले", "पैरों", "सहमति",
    "से", "को", "का", "की", "के", "में", "पर", "ने", "और", "या", "तथा", "एवं", "है", "हैं", "था", "थी", "थे",
    "हुआ", "हुई", "रहा", "रही", "रहे", "गया", "गई", "किया", "दी", "बाद", "पहले",
})
_ID_FALSE_RX = re.compile(
    r"(?i)^(?:\d{1,5}|\d{1,3}/\d{1,3}|[A-Z]\d{2}(?:\.\d+)?|[A-Z]{2}\d{3}[A-Z]|"
    r"\d{3,5}-\d|\d+(?:\.\d+)?(?:mg|mcg|ug|gm|g|ml|iu|yrs?|days?)?)$"
)

_ENTITY_OUTPUT_MAP = {
    "PER": "PATIENT_NAME",
    "REL": "RELATIVE_NAME",
    "LOC": "LOCALITY",
    "ID": "MRN",
}


def tokenize_with_spans(text: str) -> list[tuple[int, int, str]]:
    """Tokenize text while preserving character spans and atomic shield/placeholder tokens."""
    return [(m.start(), m.end(), m.group(0)) for m in _TOK_RX.finditer(text)]


def _word_shape(tok: str) -> str:
    if tok.startswith("\x00SHIELD_"):
        return "SHIELD"
    if tok.startswith("[") and tok.endswith("]"):
        return "TAG"
    if _DEVA_RX.search(tok):
        return "DEVA"
    if tok.isdigit():
        return f"D{min(len(tok), 4)}"
    if any(ch.isdigit() for ch in tok) and any(ch.isalpha() for ch in tok):
        return "ALNUM_ID"
    if any(ch.isdigit() for ch in tok):
        return "NUM_MIX"
    letters = tok.replace("-", "").replace("'", "").replace("’", "")
    if letters and letters.isalpha():
        if len(letters) == 1:
            return "INIT_U" if letters.isupper() else "INIT_L"
        if letters.isupper():
            return "UPPER"
        if letters[0].isupper() and letters[1:].islower():
            return "TITLE"
        if letters.islower():
            return "LOWER"
        return "MIXED"
    return f"P:{tok[:2]}"


def _id_skeleton(tok: str) -> str:
    """Compact structural shape for alphanumeric ID tokens without memorizing raw values."""
    out: list[str] = []
    for ch in tok[:20]:
        if ch.isalpha():
            c = "A"
        elif ch.isdigit():
            c = "0"
        else:
            c = ch
        if not out or out[-1] != c or c in "-/_#.":
            out.append(c)
    return "".join(out)


def _single_token_feats(tok: str) -> dict[str, Any]:
    norm_tok = fold_homoglyphs(tok) if not tok.isascii() else tok
    shape = _word_shape(norm_tok)
    low = norm_tok.lower()
    low_clean = low.rstrip(".")
    # Avoid memorizing open-class names across ANY case (TITLE, UPPER, LOWER, MIXED, DEVA).
    # Closed functional/clinical/org words retain their cue identity cleanly.
    closed_tok = (
        low_clean in _NON_PERSON_STOPWORDS
        or low_clean in _SALUTATION_WORDS
        or low_clean in _ROLE_PREFIX_STOPWORDS
        or low_clean in _REL_CUES
        or low_clean in _PREPOSITION_START_WORDS
        or low_clean in _PLACE_ORG_WORDS
        or low_clean in _CLINICAL_CUE_WORDS
        or low in _HINDI_CASE_WORDS
    )
    if shape == "ALNUM_ID":
        vocab_key = f"_ID:{_id_skeleton(norm_tok)}"
    elif closed_tok or shape in ("SHIELD", "TAG") or shape.startswith(("D", "P:", "NUM_")):
        vocab_key = low[:32]
    else:
        vocab_key = "_NAME_TOK"

    feats: dict[str, Any] = {
        "w.cue": vocab_key,
        "shape": shape,
        "len_b": str(min(len(norm_tok), 12)),
    }
    if shape not in ("SHIELD", "TAG"):
        feats["is_deva"] = shape == "DEVA"
        feats["is_alpha"] = shape in ("TITLE", "LOWER", "UPPER", "MIXED", "INIT_U", "INIT_L", "DEVA")
        feats["is_initial"] = shape in ("INIT_U", "INIT_L")
        feats["is_title"] = shape == "TITLE"
        feats["is_upper"] = shape == "UPPER"
        feats["is_d4"] = shape == "D4"
        feats["has_dig"] = any(ch.isdigit() for ch in norm_tok)
        feats["has_hyph"] = "-" in norm_tok
        feats["has_apos"] = "'" in norm_tok or "’" in norm_tok
        feats["is_sal"] = low_clean in _SALUTATION_WORDS
        feats["is_role"] = low_clean in _ROLE_PREFIX_STOPWORDS
        feats["is_rel"] = low_clean in _REL_CUES
        feats["is_clin"] = low_clean in _CLINICAL_CUE_WORDS
        feats["is_prep"] = low_clean in _PREPOSITION_START_WORDS
        feats["is_place"] = low_clean in _PLACE_ORG_WORDS
        feats["is_stop"] = low_clean in _NON_PERSON_STOPWORDS
        feats["is_hi_case"] = low in _HINDI_CASE_WORDS
        # Soft clinical morphology signals (lets CRF weigh suffix + context without blocking Ranvir/Sangram)
        feats["has_chem_sfx"] = bool(_CLINICAL_CHEM_SUFFIX_RX.search(low_clean))
        feats["has_strict_med"] = bool(_STRICT_MED_SUFFIX_RX.search(low_clean))
        phon = phonetic_normalize_token(low_clean)
        if phon and len(phon) >= 2 and not shape.startswith(("D", "P:")):
            feats["phon_p2"] = phon[:2]
        if len(low) >= 2 and not shape.startswith(("D", "P:")):
            feats["p2"] = low[:2]
            feats["s2"] = low[-2:]
        if len(low) >= 3 and not shape.startswith(("D", "P:")):
            feats["p3"] = low[:3]
            feats["s3"] = low[-3:]
        if len(low) >= 4 and not shape.startswith(("D", "P:")):
            feats["s4"] = low[-4:]
    return feats


def extract_sentence_features(tokens: list[str]) -> list[dict[str, Any]]:
    """Build anti-overfit pycrfsuite feature dicts for a tokenized sentence/line."""
    n = len(tokens)
    base = [_single_token_feats(t) for t in tokens]
    sent_feats: list[dict[str, Any]] = []
    for i, cur in enumerate(base):
        f: dict[str, Any] = {"bias": 1.0}
        for k, v in cur.items():
            f[f"c.{k}"] = v
        if i == 0:
            f["BOS"] = True
        if i == n - 1:
            f["EOS"] = True
        for offset, tag in ((-2, "-2"), (-1, "-1"), (1, "+1"), (2, "+2")):
            j = i + offset
            if 0 <= j < n:
                nb = base[j]
                f[f"{tag}.cue"] = nb["w.cue"]
                f[f"{tag}.shape"] = nb["shape"]
                for bkey in ("is_sal", "is_role", "is_rel", "is_clin", "is_prep", "is_place", "is_stop", "is_hi_case", "is_initial", "has_dig"):
                    if bkey in nb:
                        f[f"{tag}.{bkey}"] = nb[bkey]
        if i > 0:
            f["sh_prev"] = f"{base[i - 1]['shape']}|{cur['shape']}"
            f["al_prev"] = f"{int(base[i - 1].get('is_alpha', False))}|{int(cur.get('is_alpha', False))}"
        if i + 1 < n:
            f["sh_next"] = f"{cur['shape']}|{base[i + 1]['shape']}"
            f["al_next"] = f"{int(cur.get('is_alpha', False))}|{int(base[i + 1].get('is_alpha', False))}"
        if 0 < i < n - 1:
            f["sh_tri"] = f"{base[i - 1]['shape']}|{cur['shape']}|{base[i + 1]['shape']}"
            # Dotted-initial pattern: "K" followed by "."
            if cur.get("is_initial") and tokens[i + 1] == ".":
                f["init_dot"] = True
        sent_feats.append(f)
    return sent_feats


def _get_tagger() -> Any:
    global _TAGGER_CACHE, _TAGGER_LOADED
    if _TAGGER_LOADED:
        return _TAGGER_CACHE
    with _TAGGER_LOCK:
        if _TAGGER_LOADED:
            return _TAGGER_CACHE
        if pycrfsuite is not None and _MODEL_PATH.is_file():
            try:
                tagger = pycrfsuite.Tagger()
                tagger.open(str(_MODEL_PATH))
                _TAGGER_CACHE = tagger
            except Exception:
                _TAGGER_CACHE = None
        _TAGGER_LOADED = True
        return _TAGGER_CACHE


def _reset_tagger_cache() -> None:
    """Reset cached CRF handle after retraining."""
    global _TAGGER_CACHE, _TAGGER_LOADED
    with _TAGGER_LOCK:
        _TAGGER_CACHE = None
        _TAGGER_LOADED = False


def _is_valid_person_span(span_tokens: list[str], raw_span: str, avg_prob: float) -> bool:
    """Precision guard rejecting medical nouns, stopwords, tags, and noisy short fragments."""
    if not raw_span or "\x00SHIELD_" in raw_span or "[" in raw_span or "]" in raw_span:
        return False
    if re.search(r"\d", raw_span):
        return False
    words = [t.strip(".,;:!?\"'()-") for t in span_tokens if re.search(r"[^\W_]", t)]
    if not words or len(words) > 5:
        return False

    meaningful: list[str] = []
    initials = 0
    for w in words:
        low = w.casefold()
        if low in _NON_PERSON_STOPWORDS or low in _SALUTATION_WORDS or low in _ROLE_PREFIX_STOPWORDS or low in _REL_CUES:
            return False
        if _STRICT_MED_SUFFIX_RX.search(low):
            return False
        if len(w) == 1 and w.isascii() and w.isalpha():
            initials += 1
            continue
        meaningful.append(w)

    if not meaningful:
        return False
    if len(meaningful) == 1 and initials == 0:
        tok = meaningful[0]
        if len(tok) < 3:
            return False
        if tok.isascii() and not (tok[0].isupper() or tok.isupper()):
            return False
        if avg_prob < 0.85:
            return False
    else:
        # Allow lowercase multi-word names ONLY with stronger CRF confidence (>= 0.84)
        if all(w.isascii() and w.islower() for w in meaningful) and avg_prob < 0.84:
            return False
    return True


def _is_valid_loc_span(span_tokens: list[str], raw_span: str, avg_prob: float) -> bool:
    if not raw_span or "\x00SHIELD_" in raw_span or "[" in raw_span or "]" in raw_span:
        return False
    if len(raw_span.strip()) < 4 or avg_prob < 0.82:
        return False
    words = [t.strip(".,;:!?\"'()-") for t in span_tokens if re.search(r"[^\W_]", t)]
    if not words or len(words) > 7:
        return False
    non_stop = [w for w in words if w.casefold() not in _NON_PERSON_STOPWORDS and not w.isdigit()]
    if not non_stop:
        return False
    if any(_STRICT_MED_SUFFIX_RX.search(w.casefold()) for w in non_stop):
        return False
    if all(w.isascii() and w.islower() for w in words) and len(words) == 1:
        return False
    return True


def _is_valid_id_span(span_tokens: list[str], raw_span: str, avg_prob: float) -> bool:
    s = raw_span.strip()
    if not s or "\x00SHIELD_" in s or "[" in s or "]" in s:
        return False
    if len(s) < 5 or avg_prob < 0.84 or not re.search(r"\d", s):
        return False
    if _ID_FALSE_RX.fullmatch(s):
        return False
    return True


@lru_cache(maxsize=8192)
def _tag_line_cached(line: str, min_prob: float) -> tuple[tuple[int, int, str], ...]:
    tagger = _get_tagger()
    if tagger is None:
        return ()
    tok_spans = tokenize_with_spans(line)
    if len(tok_spans) < 2:
        return ()
    toks = [t for _, _, t in tok_spans]
    feats = extract_sentence_features(toks)
    with _TAGGER_LOCK:
        tagger.set(feats)
        labels = tagger.tag()
        probs = [tagger.marginal(lbl, idx) if lbl != "O" else 0.0 for idx, lbl in enumerate(labels)]

    spans: list[tuple[int, int, str]] = []
    i = 0
    while i < len(labels):
        lbl = labels[i]
        if lbl.startswith("B-") and probs[i] >= min_prob:
            ent = lbl[2:]
            out_kind = _ENTITY_OUTPUT_MAP.get(ent)
            j = i + 1
            while j < len(labels) and labels[j] == f"I-{ent}" and probs[j] >= (min_prob - 0.14):
                j += 1
            if out_kind is not None:
                span_s = tok_spans[i][0]
                span_e = tok_spans[j - 1][1]
                span_toks = toks[i:j]
                avg_p = sum(probs[i:j]) / (j - i)
                raw_sub = line[span_s:span_e]
                if avg_p >= min_prob:
                    ok = False
                    if ent in ("PER", "REL"):
                        ok = _is_valid_person_span(span_toks, raw_sub, avg_p)
                    elif ent == "LOC":
                        ok = _is_valid_loc_span(span_toks, raw_sub, avg_p)
                    elif ent == "ID":
                        ok = _is_valid_id_span(span_toks, raw_sub, avg_p)
                    if ok:
                        spans.append((span_s, span_e, out_kind))
            i = j
        else:
            i += 1
    return tuple(spans)


def tag_unanchored_person_spans(text: str, min_prob: float = 0.78) -> list[tuple[int, int, str]]:
    """Return high-confidence unanchored (start, end, ENTITY_KIND) spans via offline CRF."""
    if _get_tagger() is None or not text or len(text.strip()) < 4 or (" " not in text and "\t" not in text):
        return []

    spans: list[tuple[int, int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped and len(stripped) >= 4 and (" " in stripped or "\t" in stripped):
            for s, e, kind in _tag_line_cached(line, min_prob):
                spans.append((offset + s, offset + e, kind))
        offset += len(line)
    return spans


tag_unanchored_pii_spans = tag_unanchored_person_spans
