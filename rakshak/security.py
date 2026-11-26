"""Security, at-rest encryption, PII-free logging, and deterministic export assembly.

Enforces:
  * Ephemeral AES-256-GCM encryption for quarantined or staged raw payloads at rest
  * Strict PII-free counter-only logging and safe exception-reason sanitization
  * Cross-checked honest report.json (raksha-report/1) and records.jsonl export
  * Secret hygiene guard ensuring RAKSHA_SECRET never reaches outputs or logs
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_MAGIC = b"RKSHGCM1"
_SAFE_REASON_RE = re.compile(r"^[a-z][a-z0-9_]{1,48}$")


class EphemeralVault:
    """In-memory-keyed AES-256-GCM vault for encrypting raw/quarantined artifacts at rest.

    The key lives strictly in process memory for the duration of a run and is
    never written to disk or derived from RAKSHA_SECRET.
    """

    def __init__(self) -> None:
        self._key = AESGCM.generate_key(bit_length=256)
        self._aesgcm = AESGCM(self._key)

    def seal(self, plaintext: bytes, aad: bytes = b"") -> bytes:
        """Encrypt bytes with a fresh 96-bit nonce and optional associated data."""
        nonce = os.urandom(12)
        ct = self._aesgcm.encrypt(nonce, plaintext, aad)
        return _MAGIC + nonce + ct

    def unseal(self, blob: bytes, aad: bytes = b"") -> bytes:
        """Decrypt a sealed blob (used in verification/recovery tests)."""
        if not blob.startswith(_MAGIC) or len(blob) < len(_MAGIC) + 12 + 16:
            raise ValueError("invalid_vault_blob")
        nonce = blob[len(_MAGIC) : len(_MAGIC) + 12]
        ct = blob[len(_MAGIC) + 12 :]
        return self._aesgcm.decrypt(nonce, ct, aad)

    def quarantine_file(self, quarantine_dir: Path, input_sha256: str, raw_bytes: bytes) -> Path:
        """Persist quarantined raw bytes encrypted at rest under a hash-only filename."""
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        dest = quarantine_dir / f"{input_sha256}.enc"
        dest.write_bytes(self.seal(raw_bytes, aad=input_sha256.encode("ascii")))
        return dest


def safe_error_reason(exc: Exception) -> str:
    """Return a PII-free, content-free error reason code for report.json."""
    if isinstance(exc, ValueError):
        msg = str(exc).strip()
        if _SAFE_REASON_RE.match(msg):
            return msg
    return type(exc).__name__


def peak_rss_mb() -> float:
    """Return peak resident set size in MB (Linux/macOS resource, Windows GetProcessMemoryInfo)."""
    try:
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(r / 1024 if sys.platform != "darwin" else r / 1048576, 1)
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class _PMC(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            pmc = _PMC()
            pmc.cb = ctypes.sizeof(_PMC)
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            hproc = k32.GetCurrentProcess()
            get_mem = ctypes.windll.psapi.GetProcessMemoryInfo
            get_mem.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
            if get_mem(hproc, ctypes.byref(pmc), pmc.cb):
                return round(pmc.PeakWorkingSetSize / (1024 * 1024), 1)
        except Exception:
            pass
    return 0.0


def assert_no_secret_in_bytes(secret: bytes, payload: bytes, label: str = "output") -> None:
    """Fail fast if a non-empty RAKSHA_SECRET ever appears inside serialized outputs."""
    if secret and len(secret) >= 4 and secret in payload:
        raise RuntimeError(f"secret_leak_prevented:{label}")


def sha256_hex(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_release_manifests(
    out_dir: Path,
    file_entries: list[dict[str, Any]],
    records_out: list[dict[str, Any]],
    identifiers_detected: int,
    records_generalized: int,
    records_suppressed: int,
    records_quarantined: int,
    elapsed_seconds: float,
    secret: bytes,
    entity_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Deterministically write records.jsonl and honest, cross-checked report.json."""
    exported_outputs = {
        str(e["output"])
        for e in file_entries
        if e.get("status") == "sanitized" and e.get("output")
    }
    valid_records = [r for r in records_out if r.get("file") in exported_outputs]

    records_bytes = (
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in valid_records)
    ).encode("utf-8")
    assert_no_secret_in_bytes(secret, records_bytes, "records.jsonl")
    (out_dir / "records.jsonl").write_bytes(records_bytes)

    n_total = len(file_entries)
    n_san = sum(1 for e in file_entries if e.get("status") == "sanitized")
    n_quar = sum(1 for e in file_entries if e.get("status") == "quarantined")
    n_linked = sum(1 for r in valid_records if r.get("patient") is not None)

    report: dict[str, Any] = {
        "schema": "raksha-report/1",
        "files": file_entries,
        "counts": {
            "files_total": n_total,
            "files_sanitized": n_san,
            "files_quarantined": n_quar,
            "records_processed": len(valid_records),
            "identifiers_detected": int(identifiers_detected),
            "records_linked": n_linked,
            "records_generalized": int(records_generalized),
            "records_suppressed": int(records_suppressed),
            "records_quarantined": int(records_quarantined),
        },
        "timing": {"seconds": round(float(elapsed_seconds), 2)},
        "resources": {"peak_rss_mb": peak_rss_mb()},
    }
    if entity_counts:
        report["entities"] = dict(sorted(entity_counts.items()))

    report_bytes = (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    assert_no_secret_in_bytes(secret, report_bytes, "report.json")
    (out_dir / "report.json").write_bytes(report_bytes)
    return report


def log_summary(report: dict[str, Any]) -> None:
    """Emit strictly PII-free integer counters to stdout."""
    c = report["counts"]
    print(
        f"files {c['files_total']} sanitized {c['files_sanitized']} "
        f"quarantined {c['files_quarantined']} records {c['records_processed']} "
        f"linked {c['records_linked']} generalized {c['records_generalized']}"
    )
