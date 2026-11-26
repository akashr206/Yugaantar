"""Data ingestion, strict decoding, content sniffing, and safe quarantine."""
from __future__ import annotations

import base64
import binascii
import csv
import hashlib
import hmac
import io
import json
import re
import unicodedata
import urllib.parse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rakshak.translit import devanagari_to_latin, normalize_digits, phonetic_normalize_token

ALLOWED_EXTENSIONS = frozenset({".csv", ".tsv", ".json", ".txt"})
MAX_JSON_DEPTH = 64

# Undefined bytes in Windows-1252 and binary/control guards
_CP1252_UNDEFINED = frozenset({0x81, 0x8D, 0x8F, 0x90, 0x9D})
_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_ZW_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")

# Generic filename PII detection patterns (zero hardcoded dev hospital names)
_FN_PHONE_RX = re.compile(r"(?<!\d)(?:\+?91[-_]?)?0?[6-9]\d{9}(?!\d)")
_FN_ABHA_RX = re.compile(r"(?<!\d)91[-_]?\d{4}[-_]?\d{4}[-_]?\d{4}(?!\d)")
_FN_AADHAAR_RX = re.compile(r"(?<!\d)\d{4}[-_]?\d{4}[-_]?\d{4}(?!\d)")
_FN_PAN_RX = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{5}\d{4}[A-Za-z](?![A-Za-z0-9])")
_FN_EMAIL_RX = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|[\w.+-]+@(?:abdm|sbx|ndhm|ok\w+|ybl|ibl|axl|paytm|upi)\b", re.I)

_FN_MULTI_ID_RX = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{2,6}[-_]\d{2,4}(?:[-_]\d{2,})+(?![A-Za-z0-9])")
_FN_COMPACT_ID_RX = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{2,6}\d{5,}(?![A-Za-z0-9])")
_FN_HYPHEN_ID_RX = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{2,6})([-_])(\d{5,})(?![A-Za-z0-9])")
_FN_LABELED_ID_RX = re.compile(r"(?i)(?<![A-Za-z0-9])(?:uhid|mrn|ipd|pid|uid|adm)[-_]?\d{2,}(?![A-Za-z0-9])")
_ID_PREFIX_WORDS = frozenset({"uhid", "mrn", "ipd", "opd", "pid", "uid", "adm", "reg", "pat", "id", "bl"})

# Universal non-PII file/medical/structural tokens (no dataset-specific hospital or lab names)
_UNIVERSAL_GENERIC_TOKENS = frozenset(
    {
        "registration",
        "reg",
        "legacy",
        "export",
        "exports",
        "patients",
        "patient",
        "claims",
        "opd",
        "ipd",
        "ds",
        "rx",
        "nursing",
        "visitor",
        "log",
        "logs",
        "ward",
        "wards",
        "report",
        "reports",
        "whatsapp",
        "forward",
        "bulk",
        "lab",
        "labs",
        "results",
        "bundle",
        "management",
        "protocol",
        "drug",
        "drugs",
        "formulary",
        "reference",
        "ranges",
        "census",
        "readme",
        "index",
        "vitals",
        "sheet",
        "helpdesk",
        "tickets",
        "scanned",
        "consent",
        "form",
        "ambulance",
        "discharge",
        "summary",
        "note",
        "notes",
        "frag",
        "fragmentary",
        "deep",
        "truncated",
        "corrupt",
        "bytes",
        "mislabeled",
        "empty",
        "clean",
        "fhir",
        "billing",
        "misc",
        "prescription",
        "prescriptions",
        "record",
        "records",
        "data",
        "dump",
        "archive",
        "daily",
        "monthly",
        "annual",
        "copy",
        "final",
        "backup",
        "hospital",
        "clinic",
        "medical",
        "health",
        "centre",
        "center",
        "audit",
        "list",
        "table",
        "file",
        "doc",
        "docs",
        "test",
        "sample",
        "hindi",
        "english",
        "plus91",
        "visits",
        "visit",
        "admissions",
        "admission",
        "emergency",
        "pathology",
        "diagnostics",
    }
)


class QuarantineError(ValueError):
    """Raised when a file cannot be safely parsed and must be quarantined."""

    def __init__(self, reason: str) -> None:
        if not reason.isidentifier():
            reason = "quarantined"
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class IngestResult:
    """Result of ingesting and sniffing a single input file."""

    file_rel: str
    sha256: str
    ext: str
    format_type: str  # "csv" | "tsv" | "json" | "fhir" | "txt"
    text: str
    parsed: Any
    hospital_hint: str
    clean_rel: str
    filename_renamed: bool


_FACILITY_HINT_RX = re.compile(
    r"(?i)\b(?:hospital|clinic|sadan|seva|pathology|lab|labs|diagnostics|medical|centre|center|"
    r"health|trust|charitable|nursing\s+home)\b|अस्पताल|चिकित्सालय|सदन"
)
_B64_SEG_RX = re.compile(r"^[A-Za-z0-9+/_-]{16,}={0,2}$")


def _slugify_facility(raw: str) -> str:
    """Normalize a facility/hospital name into a compact lowercase identifier."""
    norm = normalize_digits(raw).strip().casefold()
    # Keep portion before trailing city/subtitle separator when present
    head = re.split(r"[,|\u2014]+|\s+-\s+", norm, maxsplit=1)[0].strip()
    slug = re.sub(r"[^a-z0-9\u0900-\u097F]+", "_", head or norm).strip("_")
    return slug


def infer_hospital_hint(
    rel: str,
    format_type: str = "",
    parsed: Any = None,
    text: str = "",
) -> str:
    """Infer a generic facility slug from document metadata, header banner, or path prefix."""
    if format_type in ("json", "fhir") and isinstance(parsed, dict):
        meta = parsed.get("_export_meta")
        if isinstance(meta, dict):
            for key in ("hospital", "facility", "organization", "institution"):
                val = meta.get(key)
                if isinstance(val, str) and val.strip():
                    return _slugify_facility(val)
        for key in ("hospital", "facility", "organization", "institution"):
            val = parsed.get(key)
            if isinstance(val, str) and val.strip():
                return _slugify_facility(val)
        if parsed.get("resourceType") == "Bundle" and isinstance(parsed.get("entry"), list):
            for ent in parsed["entry"]:
                res = ent.get("resource") if isinstance(ent, dict) else None
                if isinstance(res, dict) and res.get("resourceType") == "Organization":
                    name = res.get("name")
                    if isinstance(name, str) and name.strip():
                        return _slugify_facility(name)

    if text:
        lines = [ln.strip() for ln in text.splitlines()[:6] if ln.strip()]
        for ln in lines:
            m = re.match(r"(?i)^(?:hospital|facility|clinic|lab|organization)\s*[:\-]\s*(.+)$", ln)
            if m and len(m.group(1).strip()) >= 3:
                return _slugify_facility(m.group(1))
        if lines and ":" not in lines[0] and _FACILITY_HINT_RX.search(lines[0]):
            return _slugify_facility(lines[0])

    # Fallback: leading non-generic alpha prefix in filename (e.g. "<facility>_opd_2026.json")
    stem_parts = [p for p in re.split(r"[^A-Za-z0-9]+", Path(rel).stem) if p]
    for tok in stem_parts[:2]:
        low = tok.lower()
        if len(low) >= 3 and low.isalpha() and low not in _UNIVERSAL_GENERIC_TOKENS:
            return low
    return ""


def _is_valid_cp1252_text(raw: bytes) -> bool:
    """Reject binary garbage, mid-stream BOMs, and corrupt UTF-8 fragments."""
    if b"\x00" in raw or b"\xff\xfe" in raw or b"\xfe\xff" in raw:
        return False
    for b in raw:
        if b in _CP1252_UNDEFINED or b in (0xFE, 0xFF):
            return False
        if b < 0x20 and b not in (0x09, 0x0A, 0x0D):
            return False
    # Reject obvious UTF-8 multi-byte mojibake fragments (e.g., 0xC3 followed by ASCII punctuation)
    if re.search(rb"[\xc2-\xdf][\x00-\x7f]", raw):
        return False
    return True


def decode_strict(raw: bytes) -> str | None:
    """Decode UTF-8-BOM, UTF-16, strict UTF-8, or validated cp1252; else return None."""
    if raw.startswith(b"\xef\xbb\xbf"):
        try:
            return raw[3:].decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return None
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            text = raw.decode("utf-16", errors="strict")
            if "\x00" in text:
                return None
            return text
        except UnicodeDecodeError:
            return None
    try:
        text = raw.decode("utf-8", errors="strict")
        if any(ord(ch) < 0x20 and ch not in ("\t", "\n", "\r", "\f") for ch in text):
            return None
        return text
    except UnicodeDecodeError:
        pass

    if not _is_valid_cp1252_text(raw):
        return None
    try:
        return raw.decode("cp1252", errors="strict")
    except UnicodeDecodeError:
        return None


def _iterative_json_depth(obj: Any) -> int:
    """Compute maximum JSON container nesting depth iteratively (stack-safe)."""
    max_depth = 0
    stack: list[tuple[Any, int]] = [(obj, 1)]
    while stack:
        cur, depth = stack.pop()
        if isinstance(cur, dict):
            if depth > max_depth:
                max_depth = depth
            if depth > MAX_JSON_DEPTH:
                return depth
            for v in cur.values():
                if isinstance(v, (dict, list)):
                    stack.append((v, depth + 1))
        elif isinstance(cur, list):
            if depth > max_depth:
                max_depth = depth
            if depth > MAX_JSON_DEPTH:
                return depth
            for v in cur:
                if isinstance(v, (dict, list)):
                    stack.append((v, depth + 1))
    return max_depth
 

def _decode_segment_variants(stem: str) -> list[str]:
    """Produce NFKC/Indic-digit/URL/base64-decoded variants of a path segment."""
    norm = normalize_digits(_ZW_RE.sub("", unicodedata.normalize("NFKC", stem)))
    variants: list[str] = [norm]
    if "%" in norm:
        unquoted = normalize_digits(_ZW_RE.sub("", unicodedata.normalize("NFKC", urllib.parse.unquote(norm))))
        if unquoted != norm:
            variants.append(unquoted)
    for tok in re.split(r"[^A-Za-z0-9+/=_-]+", norm):
        if len(tok) < 16 or not _B64_SEG_RX.match(tok):
            continue
        padded = tok + "=" * (-len(tok) % 4)
        for decoder in (
            lambda b: base64.b64decode(b, validate=True),
            base64.urlsafe_b64decode,
        ):
            try:
                raw = decoder(padded)
                decoded = raw.decode("utf-8", errors="strict")
                if decoded and all(ord(c) >= 0x20 or c in "\t\n\r" for c in decoded):
                    variants.append(normalize_digits(decoded))
                    break
            except (binascii.Error, UnicodeDecodeError, ValueError):
                continue
    return variants


def compute_corpus_common_tokens(rel_paths: list[str]) -> frozenset[str]:
    """Find structural/facility tokens repeated across multiple files in the input tree."""
    if len(rel_paths) < 5:
        return frozenset()
    df: Counter[str] = Counter()
    for rel in rel_paths:
        seen_in_file: set[str] = set()
        for seg in Path(rel).parts:
            stem = Path(seg).stem if seg == Path(rel).parts[-1] else seg
            for p in re.split(r"[^A-Za-z0-9]+", stem):
                low = p.lower()
                if len(low) >= 2 and not low.isdigit():
                    seen_in_file.add(low)
        df.update(seen_in_file)
    threshold = max(3, min(8, len(rel_paths) // 25))
    return frozenset(tok for tok, cnt in df.items() if cnt >= threshold)


def _segment_has_pii(
    segment: str,
    is_filename: bool,
    known_pii_tokens: frozenset[str] = frozenset(),
    known_pii_ids: frozenset[str] = frozenset(),
    corpus_common_tokens: frozenset[str] = frozenset(),
    corpus_name_pairs: tuple[frozenset[str], ...] = (),
    file_has_pii: bool = False,
) -> bool:
    """Check whether a path component or file stem likely carries PII."""
    stem = Path(segment).stem if is_filename else segment
    for v in _decode_segment_variants(stem):
        if any(ord(ch) > 127 for ch in v):
            return True
        if (
            _FN_PHONE_RX.search(v)
            or _FN_ABHA_RX.search(v)
            or _FN_AADHAAR_RX.search(v)
            or _FN_PAN_RX.search(v)
            or _FN_EMAIL_RX.search(v)
            or _FN_MULTI_ID_RX.search(v)
            or _FN_COMPACT_ID_RX.search(v)
            or _FN_LABELED_ID_RX.search(v)
        ):
            return True
        for m in _FN_HYPHEN_ID_RX.finditer(v):
            prefix, sep = m.group(1).lower(), m.group(2)
            if prefix in _ID_PREFIX_WORDS or sep == "-" or (
                prefix not in _UNIVERSAL_GENERIC_TOKENS and prefix not in corpus_common_tokens
            ):
                return True

        if known_pii_ids:
            compact = re.sub(r"[^a-z0-9]", "", v.lower())
            if any(len(kid) >= 5 and kid in compact for kid in known_pii_ids):
                return True

        parts = [p for p in re.split(r"[^A-Za-z0-9]+", v) if p]
        if known_pii_tokens:
            for p in parts:
                if len(p) < 3 or p.isdigit():
                    continue
                low = p.lower()
                if low in _UNIVERSAL_GENERIC_TOKENS or low in corpus_common_tokens:
                    continue
                if low in known_pii_tokens or phonetic_normalize_token(low) in known_pii_tokens:
                    return True

        if corpus_name_pairs and len(parts) >= 2:
            low_parts = {
                p.lower()
                for p in parts
                if len(p) >= 3 and not p.isdigit()
            } - _UNIVERSAL_GENERIC_TOKENS - corpus_common_tokens
            if len(low_parts) >= 2:
                phon_parts = {phonetic_normalize_token(p) for p in low_parts}
                if any(pair <= low_parts or pair <= phon_parts for pair in corpus_name_pairs):
                    return True

        if is_filename and len(parts) >= 2:
            non_generic_cap = [
                p
                for p in parts
                if p[0].isupper()
                and p.lower() not in _UNIVERSAL_GENERIC_TOKENS
                and p.lower() not in corpus_common_tokens
                and not p.isdigit()
            ]
            if len(non_generic_cap) >= 2:
                return True

            if file_has_pii and corpus_common_tokens:
                rare_alpha = [
                    p
                    for p in parts
                    if len(p) >= 3
                    and not p.isdigit()
                    and p.lower() not in _UNIVERSAL_GENERIC_TOKENS
                    and p.lower() not in corpus_common_tokens
                ]
                if len(rare_alpha) >= 2:
                    return True
    return False


def sanitize_rel_path(
    rel: str,
    secret: bytes,
    known_pii_tokens: frozenset[str] = frozenset(),
    known_pii_ids: frozenset[str] = frozenset(),
    corpus_common_tokens: frozenset[str] = frozenset(),
    corpus_name_pairs: tuple[frozenset[str], ...] = (),
    file_has_pii: bool = False,
) -> tuple[str, bool]:
    """Deterministically rename any file or folder segment containing PII."""
    parts = Path(rel).parts
    if not parts:
        return rel, False

    any_renamed = False
    new_parts: list[str] = []
    for idx, seg in enumerate(parts):
        is_last = idx == len(parts) - 1
        if _segment_has_pii(
            seg,
            is_filename=is_last,
            known_pii_tokens=known_pii_tokens,
            known_pii_ids=known_pii_ids,
            corpus_common_tokens=corpus_common_tokens,
            corpus_name_pairs=corpus_name_pairs,
            file_has_pii=file_has_pii,
        ):
            any_renamed = True
            digest = hmac.new(secret, f"path|{rel}|{idx}".encode("utf-8"), hashlib.sha256).hexdigest()[:10]
            if is_last:
                ext = Path(seg).suffix.lower()
                new_parts.append(f"doc_{digest}{ext}")
            else:
                new_parts.append(f"dir_{digest}")
        else:
            new_parts.append(seg)

    return "/".join(new_parts), any_renamed


def ingest_file(path: Path, inp_root: Path, secret: bytes) -> IngestResult:
    """Read, validate, sniff, and decode a single input file or raise QuarantineError."""
    rel = path.relative_to(inp_root).as_posix()
    raw = path.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    ext = path.suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise QuarantineError("unsupported_format")
    if not raw.strip():
        raise QuarantineError("empty_file")

    text = decode_strict(raw)
    if text is None or not text.strip():
        raise QuarantineError("undecodable_text")

    clean_rel, filename_renamed = sanitize_rel_path(rel, secret)

    if ext in (".csv", ".tsv"):
        stripped = text.lstrip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                json.loads(stripped)
                raise QuarantineError("format_mismatch")
            except json.JSONDecodeError:
                if stripped.startswith("{"):
                    raise QuarantineError("format_mismatch")
        delim = "\t" if ext == ".tsv" else ","
        try:
            rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=delim))
        except csv.Error as exc:
            raise QuarantineError("csv_error") from exc
        if not rows or all(not any(c.strip() for c in r) for r in rows):
            raise QuarantineError("empty_file")
        fmt = "tsv" if ext == ".tsv" else "csv"
        parsed = (delim, rows[0], rows[1:])
        hospital_hint = infer_hospital_hint(rel, format_type=fmt, parsed=parsed, text=text)
        return IngestResult(
            file_rel=rel,
            sha256=sha256,
            ext=ext,
            format_type=fmt,
            text=text,
            parsed=parsed,
            hospital_hint=hospital_hint,
            clean_rel=clean_rel,
            filename_renamed=filename_renamed,
        )

    if ext == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise QuarantineError("JSONDecodeError") from exc
        if _iterative_json_depth(data) > MAX_JSON_DEPTH:
            raise QuarantineError("excessive_nesting")
        is_fhir = isinstance(data, dict) and data.get("resourceType") == "Bundle"
        fmt = "fhir" if is_fhir else "json"
        hospital_hint = infer_hospital_hint(rel, format_type=fmt, parsed=data, text=text)
        return IngestResult(
            file_rel=rel,
            sha256=sha256,
            ext=ext,
            format_type=fmt,
            text=text,
            parsed=data,
            hospital_hint=hospital_hint,
            clean_rel=clean_rel,
            filename_renamed=filename_renamed,
        )

    hospital_hint = infer_hospital_hint(rel, format_type="txt", parsed=text, text=text)
    return IngestResult(
        file_rel=rel,
        sha256=sha256,
        ext=ext,
        format_type="txt",
        text=text,
        parsed=text,
        hospital_hint=hospital_hint,
        clean_rel=clean_rel,
        filename_renamed=filename_renamed,
    )
