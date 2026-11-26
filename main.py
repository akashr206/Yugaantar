#!/usr/bin/env python3
"""Secret Wards privacy firewall pipeline (Enma / Rakshak).

Orchestrates two-pass deterministic sanitization:
  Pass 1: Ingest, strict decode, content-sniff, encrypted quarantine, and profile schemas -> RecordUnits
  Plan  : Cross-script identity resolution, keyed HMAC pseudonyms, and Option C adaptive k-anonymity
  Pass 2: Context-aware multilingual redaction, quasi-identifier transformation,
          FHIR R4 preservation, filename hygiene, and verified manifest export.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import time
from pathlib import Path
from typing import Callable

from rakshak.contracts import PolicyConfig, PrivacyPlan, RecordUnit
from rakshak.detector import DetectionStats
from rakshak.formats import build_record_units, render_sanitized_file
from rakshak.ingest import IngestResult, QuarantineError, ingest_file
from rakshak.planner import plan_privacy_release
from rakshak.security import (
    EphemeralVault,
    log_summary,
    safe_error_reason,
    sha256_hex,
    write_release_manifests,
)

HERE = Path(__file__).resolve().parent


def run_pipeline(
    inp: Path,
    out: Path,
    policy: PolicyConfig,
    secret: bytes,
    plan_fn: Callable[[list[RecordUnit], PolicyConfig, bytes], dict[str, PrivacyPlan]] | None = None,
) -> dict:
    """Execute the complete two-pass privacy firewall pipeline."""
    t0 = time.monotonic()
    san = out / "sanitized"
    quar_dir = out / "quarantine"
    san.mkdir(parents=True, exist_ok=True)

    vault = EphemeralVault()
    file_entries: list[dict] = []
    ingested: list[tuple[IngestResult, dict, int]] = []
    all_units: list[RecordUnit] = []
    seen_out_paths: set[str] = set()

    for path in sorted(p for p in inp.rglob("*") if p.is_file()):
        rel = path.relative_to(inp).as_posix()
        try:
            item = ingest_file(path, inp, secret)
            # Guard against rare filename collision after sanitization
            out_rel = item.clean_rel
            if out_rel in seen_out_paths:
                stem = Path(out_rel).stem
                ext = Path(out_rel).suffix
                parent = Path(out_rel).parent.as_posix()
                suffix_hash = hmac.new(secret, rel.encode("utf-8"), hashlib.sha256).hexdigest()[:6]
                out_rel = f"{parent}/{stem}_{suffix_hash}{ext}" if parent != "." else f"{stem}_{suffix_hash}{ext}"
                item = IngestResult(
                    file_rel=item.file_rel,
                    sha256=item.sha256,
                    ext=item.ext,
                    format_type=item.format_type,
                    text=item.text,
                    parsed=item.parsed,
                    hospital_hint=item.hospital_hint,
                    clean_rel=out_rel,
                    filename_renamed=True,
                )
            seen_out_paths.add(out_rel)
            units, ctx = build_record_units(item)
            entry_idx = len(file_entries)
            file_entries.append({"input_sha256": item.sha256})
            ingested.append((item, ctx, entry_idx))
            all_units.extend(units)
        except QuarantineError as exc:
            raw = path.read_bytes()
            raw_sha = sha256_hex(raw)
            vault.quarantine_file(quar_dir, raw_sha, raw)
            file_entries.append(
                {
                    "input_sha256": raw_sha,
                    "output": None,
                    "status": "quarantined",
                    "reason": exc.reason,
                }
            )
        except Exception as exc:  # noqa: BLE001 - fail-safe quarantine without leaking content
            raw = path.read_bytes()
            raw_sha = sha256_hex(raw)
            vault.quarantine_file(quar_dir, raw_sha, raw)
            file_entries.append(
                {
                    "input_sha256": raw_sha,
                    "output": None,
                    "status": "quarantined",
                    "reason": safe_error_reason(exc),
                }
            )

    planner = plan_fn or plan_privacy_release
    plans = planner(all_units, policy, secret)

    stats = DetectionStats()
    records_out: list[dict] = []
    for item, ctx, entry_idx in ingested:
        rendered = render_sanitized_file(item, ctx, plans, records_out, stats)
        dest = san / item.clean_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(rendered, encoding="utf-8", newline="")
        file_entries[entry_idx].update(output=f"sanitized/{item.clean_rel}", status="sanitized")

    n_gen = sum(1 for r in records_out if r.get("status") == "generalized")
    n_sup = sum(1 for r in records_out if r.get("status") == "suppressed")
    return write_release_manifests(
        out_dir=out,
        file_entries=file_entries,
        records_out=records_out,
        identifiers_detected=stats.identifiers,
        records_generalized=max(n_gen, stats.generalized),
        records_suppressed=n_sup,
        records_quarantined=0,
        elapsed_seconds=time.monotonic() - t0,
        secret=secret,
        entity_counts=dict(stats.by_type),
    )


def _unquote_cli_path(raw: str) -> Path:
    """Strip POSIX shlex single/double quotes when invoked through Windows cmd.exe."""
    s = raw.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1]
    return Path(s)


def main() -> int:
    ap = argparse.ArgumentParser(description="Enma / Secret Wards privacy firewall")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    policy = PolicyConfig.from_yaml(HERE / "config" / "policy.yaml")
    secret = os.environ.get("RAKSHA_SECRET", "").encode("utf-8")
    report = run_pipeline(_unquote_cli_path(args.input), _unquote_cli_path(args.output), policy, secret)
    log_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
