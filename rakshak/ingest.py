"""Data ingestion, strict decoding, content sniffing, and safe quarantine."""
from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALLOWED_EXTENSIONS = frozenset({".csv", ".tsv", ".json", ".txt"})
MAX_JSON_DEPTH = 64

# Undefined bytes in Windows-1252 and binary/control guards
_CP1252_UNDEFINED = frozenset({0x81, 0x8D, 0x8F, 0x90, 0x9D})

# Filename PII detection patterns (case-insensitive)
_FN_PHONE_RX = re.compile(r"(?<!\d)(?:\+?91[-_]?)?0?[6-9]\d{9}(?!\d)")
_FN_ID_RX = re.compile(
    r"(?i)(?:snj|jjh|dhh|uhid|ipd|ip|adm|bl|opd|mrn)[-_]?\d{2,4}(?:[-_]\d+)*|"
    r"(?<![A-Za-z0-9])[a-z]{2,5}[-_]\d{2,4}[-_]\d{3,}(?![A-Za-z0-9])|"
    r"(?<![A-Za-z0-9])[a-z]{2,5}\d{5,}(?![A-Za-z0-9])"
)
_FN_DISCHARGE_NAME_RX = re.compile(r"(?i)^discharge_summary_[a-z]+")
_FN_GENERIC_TOKENS = frozenset(
    {
        "arogya",
        "sanjeevani",
        "jeevanjyoti",
        "dhhardoi",
        "dhh",
        "snj",
        "jjh",
        "ass",
        "registration",
        "legacy",
        "export",
        "patients",
        "claims",
        "opd",
        "ds",
        "rx",
        "nursing",
        "visitor",
        "log",
        "ward",
        "ward5",
        "report",
        "whatsapp",
        "forward",
        "bulk",
        "lab",
        "city",
        "lifeline",
        "pathkind",
        "results",
        "bundle",
        "dengue",
        "management",
        "protocol",
        "drug",
        "formulary",
        "reference",
        "ranges",
        "census",
        "readme",
        "patient",
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


def infer_hospital_hint(rel: str) -> str:
    """Infer facility code from relative file path if recognizable."""
    low = rel.lower()
    if "sanjeevani" in low or "snj" in low:
        return "snj"
    if "jeevanjyoti" in low or "jjh" in low:
        return "jjh"
    if "arogya" in low or "adm" in low:
        return "ass"
    if "dhhardoi" in low or "dhh" in low:
        return "dhh"
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


def _segment_has_pii(segment: str, is_filename: bool) -> bool:
    """Check whether a path component or file stem likely carries PII."""
    stem = Path(segment).stem if is_filename else segment
    if _FN_PHONE_RX.search(stem) or _FN_ID_RX.search(stem) or _FN_DISCHARGE_NAME_RX.search(stem):
        return True
    # Check for capitalized person-name pairs like "John_Doe" not in known non-PII words
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", stem) if p]
    if is_filename and len(parts) >= 2:
        non_generic_cap = [
            p for p in parts if p[0].isupper() and p.lower() not in _FN_GENERIC_TOKENS and not p.isdigit()
        ]
        if len(non_generic_cap) >= 2:
            return True
    return False


def sanitize_rel_path(rel: str, secret: bytes) -> tuple[str, bool]:
    """Deterministically rename any file or folder segment containing PII."""
    parts = Path(rel).parts
    if not parts:
        return rel, False

    any_renamed = False
    new_parts: list[str] = []
    for idx, seg in enumerate(parts):
        is_last = idx == len(parts) - 1
        if _segment_has_pii(seg, is_filename=is_last):
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

    hospital_hint = infer_hospital_hint(rel)
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
        return IngestResult(
            file_rel=rel,
            sha256=sha256,
            ext=ext,
            format_type="tsv" if ext == ".tsv" else "csv",
            text=text,
            parsed=(delim, rows[0], rows[1:]),
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
        return IngestResult(
            file_rel=rel,
            sha256=sha256,
            ext=ext,
            format_type="fhir" if is_fhir else "json",
            text=text,
            parsed=data,
            hospital_hint=hospital_hint,
            clean_rel=clean_rel,
            filename_renamed=filename_renamed,
        )

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
