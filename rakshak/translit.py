"""Cross-script transliteration, phonetic normalization, and ID canonicalization.

Unifies English, Devanagari Hindi, and Hinglish surface forms into canonical
Latin tokens and phonetic keys without relying on memorized dev-set names.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import jellyfish
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from rapidfuzz import fuzz

_DEVA_RE = re.compile(r"[\u0900-\u097F]")
_ZW_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff\u2060\u00ad]")
_NON_ALNUM_SPACE_RE = re.compile(r"[^\w\s]", re.UNICODE)
_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_HOMOGLYPH_TRANS = str.maketrans({
    # Cyrillic uppercase
    "\u0410": "A", "\u0412": "B", "\u0415": "E", "\u041a": "K", "\u041c": "M",
    "\u041d": "H", "\u041e": "O", "\u0420": "P", "\u0421": "C", "\u0422": "T",
    "\u0425": "X", "\u0406": "I", "\u0408": "J", "\u0405": "S",
    # Cyrillic lowercase
    "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0440": "p", "\u0441": "c",
    "\u0443": "y", "\u0445": "x", "\u0456": "i", "\u0458": "j", "\u0455": "s", "\u0475": "v",
    # Greek uppercase
    "\u0391": "A", "\u0392": "B", "\u0395": "E", "\u0396": "Z", "\u0397": "H",
    "\u0399": "I", "\u039a": "K", "\u039c": "M", "\u039d": "N", "\u039f": "O",
    "\u03a1": "P", "\u03a4": "T", "\u03a5": "Y", "\u03a7": "X",
    # Greek lowercase
    "\u03b1": "a", "\u03b5": "e", "\u03b9": "i", "\u03bd": "v", "\u03bf": "o",
    "\u03c1": "p", "\u03c4": "t", "\u03c5": "u", "\u03c7": "x",
})
_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)

# Honorifics / kinship titles stripped from person name tokens
HONORIFIC_TOKENS = frozenset({
    "ji", "sahab", "saheb", "babu", "amma", "didi", "bhaiya", "bhai",
    "behenji", "bahanji", "shri", "shrimati", "shriman", "sushri", "smt",
    "kum", "ku", "km", "sh", "mr", "mrs", "ms", "miss", "dr", "prof",
    "er", "adv", "col", "maj", "capt", "brig", "master", "baby", "late", "haji",
})

# Generic tokens that must never link two records on their own
GENERIC_NAME_TOKENS = frozenset({
    "devi", "kumar", "kumari", "singh", "kaur", "bano", "begum", "bai",
    "lal", "ram", "prasad", "nath", "mohammad",
})

# Canonical cross-phonetic aliases (subsequence contractions are also matched algorithmically below)
NAME_PREFIX_MAP = {
    "md": "mohammad",
    "mohd": "mohammad",
    "mohammed": "mohammad",
    "mohammad": "mohammad",
    "muhammad": "mohammad",
    "mhd": "mohammad",
    "sri": "shri",
    "srimati": "shrimati",
    "susri": "sushri",
}


def normalize_digits(text: str) -> str:
    """Fold NFKC, strip zero-width chars, and convert Devanagari digits to ASCII."""
    cleaned = _ZW_RE.sub("", unicodedata.normalize("NFKC", text))
    return cleaned.translate(_DEVA_DIGITS)


def fold_homoglyphs(text: str, keep_deva_digits: bool = False) -> str:
    """Strip zero-width obfuscation, normalize NFKC, fold Cyrillic/Greek homoglyphs and Latin accents."""
    if text.isascii():
        return text
    cleaned = _ZW_RE.sub("", unicodedata.normalize("NFKC", text)).translate(_HOMOGLYPH_TRANS)
    # Strip Latin combining accents while keeping Devanagari marks untouched
    if any("\u0300" <= ch <= "\u036f" or unicodedata.category(ch) == "Mn" for ch in cleaned if not ("\u0900" <= ch <= "\u097f")):
        out: list[str] = []
        for ch in cleaned:
            if "\u0900" <= ch <= "\u097f" or ord(ch) < 128:
                out.append(ch)
            else:
                dec = unicodedata.normalize("NFKD", ch)
                kept = "".join(c for c in dec if unicodedata.category(c) != "Mn")
                out.append(kept)
        cleaned = "".join(out)
    return cleaned if keep_deva_digits else cleaned.translate(_DEVA_DIGITS)


def is_valid_verhoeff(digits: str) -> bool:
    """Validate Aadhaar Verhoeff check digit for a 12-digit ASCII numeric string."""
    if len(digits) != 12 or not digits.isdigit():
        return False
    c = 0
    for idx, ch in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[idx % 8][int(ch)]]
    return c == 0


def _apply_hindi_schwa_rules(itrans_word: str) -> str:
    """Convert an ITRANS-encoded Devanagari word into natural Hindi-Latin spelling.

    Operates BEFORE lowercasing because ITRANS distinguishes short inherent 'a'
    (lowercase, often silent at word-end) from long 'A' (uppercase, always pronounced).
    """
    w = itrans_word
    # Handle nukta / retroflex flaps emitted by ITRANS before lowercasing
    w = w.replace(".Dh", "rh").replace(".D", "d").replace("~N", "n").replace("~n", "n")
    w = w.replace("N^", "n").replace("RRi", "ri").replace("RRI", "ri").replace("LLi", "li")
    w = w.replace("Ch", "chh").replace("Sh", "sh")
    # Nukta consonants: K -> kh (ख़), G -> gh (ग़), J -> z (ज़), ph/f
    w = w.replace("K", "kh").replace("G", "gh").replace("J", "z")
    # Anusvara 'M': labial assimilation (m before p/b), else n
    w = re.sub(r"M([pPbBmM])", r"m\1", w)
    w = w.replace("M", "n")
    # Chandrabindu / visarga
    w = w.replace(".n", "n").replace("H", "h")
    # Drop silent trailing inherent 'a' after a consonant (Hindi schwa deletion)
    if len(w) > 2 and w.endswith("a") and not w.endswith("A"):
        prev = w[-2]
        if prev.lower() not in "aeiou":
            w = w[:-1]
    w = w.lower()
    # Map bare ITRANS 'c' (when not part of 'ch') to 'ch'
    w = re.sub(r"c(?!h)", "ch", w)
    return w


@lru_cache(maxsize=8192)
def devanagari_to_latin(text: str) -> str:
    """Transliterate Devanagari text to natural Hindi-Latin, preserving non-Deva tokens."""
    normed = normalize_digits(text).strip()
    if not normed or not _DEVA_RE.search(normed):
        return normed.casefold()

    out_parts: list[str] = []
    for tok in re.split(r"(\s+)", normed):
        if not tok or tok.isspace():
            out_parts.append(" " if tok else "")
            continue
        if _DEVA_RE.search(tok):
            # Strip punctuation attached to Devanagari word before transliterating
            stripped = tok.strip(".,;:!?।|/\\()[]{}\"'`-")
            if not stripped:
                continue
            itrans = transliterate(stripped, sanscript.DEVANAGARI, sanscript.ITRANS)
            out_parts.append(_apply_hindi_schwa_rules(itrans))
        else:
            out_parts.append(tok.casefold())
    return " ".join("".join(out_parts).split())


@lru_cache(maxsize=8192)
def phonetic_normalize_token(token: str) -> str:
    """Normalize common Indian Latin spelling variants into a canonical form."""
    t = token.casefold().strip()
    t = re.sub(r"[^a-z0-9]", "", t)
    if not t:
        return ""
    t = NAME_PREFIX_MAP.get(t, t)
    # Normalize common vowel and consonant alternations in Indian transliteration
    t = re.sub(r"(ee|ii|ei)", "i", t)
    t = re.sub(r"(oo|uu)", "u", t)
    t = re.sub(r"aa", "a", t)
    t = re.sub(r"(au|ou)", "o", t)
    t = t.replace("ph", "f").replace("w", "v").replace("q", "k")
    # Common trailing variations: pandey / pandeya / pande, vijay / vijai
    t = re.sub(r"eya$", "e", t)
    t = re.sub(r"ey$", "e", t)
    t = re.sub(r"(ai|ay)$", "ai", t)
    # Aspirate / geminate folding for fuzzy robustness
    t = t.replace("th", "t").replace("dh", "d").replace("kh", "k").replace("gh", "g").replace("bh", "b")
    t = re.sub(r"(.)\1+", r"\1", t)
    return t


@lru_cache(maxsize=8192)
def consonant_skeleton(token: str) -> str:
    """Vowel-collapsed consonant skeleton preserving leading vowel (stops Asha vs Usha)."""
    p = phonetic_normalize_token(token)
    if not p:
        return ""
    # Fold sibilant alternation (sarma/sharma, vikas/vikash)
    p = p.replace("sh", "s")
    # Handle Hindi medial schwa deletion (e.g. akaram <-> akram, bhimapur <-> bhimpur)
    first = p[0]
    rest = re.sub(r"[aeiouy]+", "", p[1:])
    skel = first + rest
    skel = re.sub(r"(.)\1+", r"\1", skel)
    return skel


def extract_name_tokens(raw_name: str, strip_honorifics: bool = True) -> tuple[str, ...]:
    """Return canonical Latin tokens from a name in Latin, Hinglish, or Devanagari."""
    latin = devanagari_to_latin(raw_name)
    latin = _NON_ALNUM_SPACE_RE.sub(" ", latin)
    tokens: list[str] = []
    for tok in latin.split():
        if not tok or tok.isdigit():
            continue
        expanded = NAME_PREFIX_MAP.get(tok, tok)
        if strip_honorifics and expanded in HONORIFIC_TOKENS:
            continue
        tokens.append(expanded)
    return tuple(tokens)


def phonetic_name_tokens(raw_name: str) -> tuple[str, ...]:
    """Return phonetic-normalized tokens for cross-script / spelling-variant matching."""
    toks = extract_name_tokens(raw_name, strip_honorifics=True)
    return tuple(phonetic_normalize_token(t) for t in toks if phonetic_normalize_token(t))


def _is_ordered_subseq(short: str, full: str) -> bool:
    """Return True if all chars of `short` appear in order inside `full`."""
    it = iter(full)
    return all(ch in it for ch in short)


def _is_token_abbreviation(a: str, b: str) -> bool:
    """Detect Indian contractions (sk<->shaikh, kr<->kumar, pd<->prasad), truncations, or initials."""
    if not a or not b or len(a) == len(b):
        return False
    short, full = (a, b) if len(a) < len(b) else (b, a)
    if short[0] != full[0]:
        return False
    if len(short) == 1:
        return len(full) >= 2
    if len(short) >= 3 and full.startswith(short):
        return True
    if len(short) <= 5:
        return _is_ordered_subseq(short, full)
    return False


def has_compatible_abbrev(tokens_a: tuple[str, ...], tokens_b: tuple[str, ...]) -> bool:
    """Return True if two multi-token names agree symmetrically up to initials/truncations/contractions."""
    pa = [phonetic_normalize_token(t) for t in tokens_a if t]
    pb = [phonetic_normalize_token(t) for t in tokens_b if t]
    if len(pa) < 2 or len(pb) < 2:
        return False
    sa = [consonant_skeleton(t) for t in tokens_a if t]
    sb = [consonant_skeleton(t) for t in tokens_b if t]

    def _pair_ok(x: str, y: str, sx: str, sy: str) -> tuple[bool, bool]:
        if x == y or (sx == sy and len(sx) >= 2):
            return True, len(x) >= 3 and len(y) >= 3
        if jellyfish.jaro_winkler_similarity(x, y) >= 0.85 or fuzz.ratio(x, y) >= 85:
            return True, True
        if _is_token_abbreviation(x, y):
            return True, False
        return False, False

    def _side_covered(p1: list[str], s1: list[str], p2: list[str], s2: list[str]) -> bool:
        anchor = False
        for i, x in enumerate(p1):
            matched = False
            for j, y in enumerate(p2):
                ok, is_full = _pair_ok(x, y, s1[i], s2[j])
                if ok:
                    matched = True
                    if is_full:
                        anchor = True
                    break
            if not matched and len(x) > 1 and x not in GENERIC_NAME_TOKENS:
                return False
        return anchor

    return _side_covered(pa, sa, pb, sb) and _side_covered(pb, sb, pa, sa)


def name_tokens_match(tokens_a: tuple[str, ...], tokens_b: tuple[str, ...]) -> float:
    """Compute precision-oriented similarity in [0.0, 1.0] between two token tuples.

    Handles exact/phonetic matches, consonant-skeleton equality (e.g. akaram vs akram),
    prefix truncations, ordered-consonant contractions (sk vs shaikh, kr vs kumar),
    and high Jaro-Winkler / RapidFuzz token match, while avoiding single generic-token matches.
    """
    if not tokens_a or not tokens_b:
        return 0.0
    pa = [phonetic_normalize_token(t) for t in tokens_a if t]
    pb = [phonetic_normalize_token(t) for t in tokens_b if t]
    if not pa or not pb:
        return 0.0
    if pa == pb:
        return 1.0

    sa = [consonant_skeleton(t) for t in tokens_a if t]
    sb = [consonant_skeleton(t) for t in tokens_b if t]
    if sa == sb:
        return 0.96

    def _tok_sim(x: str, y: str, sx: str, sy: str) -> float:
        if x == y:
            return 1.0
        if sx == sy and (len(sx) >= 3 or (len(sx) >= 2 and len(pa) >= 2 and len(pb) >= 2)):
            return 0.95
        if len(x) >= 2 and len(y) >= 2 and _is_token_abbreviation(x, y):
            return 0.87
        jw = jellyfish.jaro_winkler_similarity(x, y)
        rf = fuzz.ratio(x, y) / 100.0
        return max(jw * 0.95, rf)

    if len(pa) == 1 or len(pb) == 1:
        single_p, single_s, multi_p, multi_s = (
            (pa[0], sa[0], pb, sb) if len(pa) == 1 else (pb[0], sb[0], pa, sa)
        )
        if single_p in GENERIC_NAME_TOKENS:
            return 0.0
        best = 0.0
        for mp, ms in zip(multi_p, multi_s):
            if mp in GENERIC_NAME_TOKENS:
                continue
            best = max(best, _tok_sim(single_p, mp, single_s, ms))
        return best * 0.85 if best >= 0.88 else 0.0

    # Multi-token full name: compare first+last and handle reversed order ("Shahid Mohammad")
    if len(pa) == len(pb) == 2 and set(sa) == set(sb):
        return 0.94

    # Greedy best token alignment
    first_sim = _tok_sim(pa[0], pb[0], sa[0], sb[0])
    last_sim = _tok_sim(pa[-1], pb[-1], sa[-1], sb[-1])
    if first_sim >= 0.86 and last_sim >= 0.84:
        return round((first_sim + last_sim) / 2.0, 4)

    joined_a, joined_b = " ".join(pa), " ".join(pb)
    tsr = fuzz.token_sort_ratio(joined_a, joined_b) / 100.0
    return tsr if tsr >= 0.88 else 0.0


def norm_phone(raw: str) -> str | None:
    """Normalize an Indian phone number to 10 digits starting with [6-9], or valid STD landline."""
    digits = re.sub(r"\D", "", normalize_digits(raw))
    if len(digits) == 14 and digits.startswith("0091"):
        digits = digits[4:]
    elif len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    if len(digits) == 10 and digits[0] in "6789":
        return digits
    if re.search(r"[-.\s]", raw) and re.fullmatch(r"0\d{9,10}", digits):
        return digits
    return None


def norm_aadhaar(raw: str) -> str | None:
    """Return 12-digit Aadhaar if unmasked, else None."""
    s = normalize_digits(raw)
    if "x" in s.lower() or "*" in s:
        return None
    digits = re.sub(r"\D", "", s)
    return digits if len(digits) == 12 else None


def norm_aadhaar_last4(raw: str) -> str | None:
    """Extract last 4 digits from a masked Aadhaar such as 'XXXX XXXX 5866'."""
    s = normalize_digits(raw)
    m = re.search(r"(?:(?:\([xX*]{4}\)|[xX*]{4})[\s.-]?){2}(\d{4})\b", s)
    return m.group(1) if m else None


def norm_abha(raw: str) -> str | None:
    """Return 14-digit ABHA number if valid."""
    digits = re.sub(r"\D", "", normalize_digits(raw))
    return digits if len(digits) == 14 else None


def norm_mrn(raw: str) -> tuple[str, str, str] | None:
    """Normalize MRN/UHID/IPD into (canonical_full, digit_core, hospital_prefix)."""
    s = normalize_digits(raw).strip().upper()
    if not s or len(s) < 5:
        return None
    # Strip trailing separators
    s = re.sub(r"\s+", "", s)
    digits = re.sub(r"\D", "", s)
    if len(digits) < 4:
        return None
    # Extract trailing digit core (most UHIDs end with 5-7 sequential digits)
    m_core = re.search(r"(\d{5,8})$", s)
    digit_core = m_core.group(1) if m_core else digits
    m_pref = re.match(r"^([A-Z]{2,6})", s)
    prefix = m_pref.group(1).lower() if m_pref else ""
    canon = re.sub(r"[^A-Z0-9]", "", s).lower()
    return canon, digit_core, prefix


def norm_location(raw: str) -> str:
    """Normalize village, tehsil, or district across Latin and Devanagari."""
    latin = devanagari_to_latin(raw)
    latin = _NON_ALNUM_SPACE_RE.sub(" ", latin)
    toks = [consonant_skeleton(t) for t in latin.split() if t and consonant_skeleton(t)]
    return " ".join(toks)


def norm_district(raw: str) -> str:
    """Normalize district name (e.g. 'Rae Bareli' == 'raebareli')."""
    latin = devanagari_to_latin(raw)
    latin = _NON_ALNUM_SPACE_RE.sub("", latin)
    return consonant_skeleton(latin)
