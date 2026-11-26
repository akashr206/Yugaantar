#!/usr/bin/env python3
"""Secret Wards privacy firewall pipeline (Enma / Rakshak).

Orchestrates two-pass deterministic sanitization:
  Pass 1: Ingest, strict decode, content-sniff, quarantine, and profile schemas -> RecordUnits
  Plan  : Invoke Person 2 linkage & privacy planner (falls back safely when stubbed)
  Pass 2: Context-aware multilingual redaction, quasi-identifier transformation,
          FHIR R4 preservation, filename hygiene, and report/records serialization.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

from rakshak.contracts import PolicyConfig, PrivacyPlan, RecordUnit
from rakshak.detector import DetectionStats
from rakshak.formats import build_record_units, render_sanitized_file
from rakshak.ingest import IngestResult, QuarantineError, ingest_file

HERE = Path(__file__).resolve().parent


def _default_date_shift(key: str, secret: bytes, max_days: int) -> int:
    digest = hmac.new(secret, b"shift|" + key.encode("utf-8"), hashlib.sha256).digest()
    val = int.from_bytes(digest[:4], "big") % (2 * max_days) - max_days
    return val if val != 0 else 1


def _fallback_privacy_plans(
    units: list[RecordUnit],
    policy: PolicyConfig,
    secret: bytes,
) -> dict[str, PrivacyPlan]:
    """Person-1 standalone fallback until Person 2 linkage + privacy modules plug in."""
    plans: dict[str, PrivacyPlan] = {}
    for u in units:
        plans[u.unit_id] = PrivacyPlan(
            unit_id=u.unit_id,
            cluster_key=None,
            pseudonym=None,
            date_shift_days=_default_date_shift(u.unit_id, secret, policy.date_shift_max_days),
            age_band=None,
            keep_district=True,
            keep_pin_prefix=True,
            generalization_level="standard",
        )
    return plans


def _resolve_planner() -> Callable[[list[RecordUnit], PolicyConfig, bytes], dict[str, PrivacyPlan]]:
    """Use Person 2's planner if implemented, otherwise fallback safely."""
    try:
        from rakshak import privacy as p2_privacy

        fn = getattr(p2_privacy, "build_privacy_plans", None)
        if callable(fn):
            return fn
    except Exception:  # noqa: BLE001
        pass
    return _fallback_privacy_plans


def peak_rss_mb() -> float:
    try:
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(r / 1024 if sys.platform != "darwin" else r / 1048576, 1)
    except Exception:  # noqa: BLE001
        return 0.0


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
    san.mkdir(parents=True, exist_ok=True)

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
            raw_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            file_entries.append(
                {
                    "input_sha256": raw_sha,
                    "output": None,
                    "status": "quarantined",
                    "reason": exc.reason,
                }
            )
        except Exception as exc:  # noqa: BLE001 - fail-safe quarantine without leaking content
            raw_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            file_entries.append(
                {
                    "input_sha256": raw_sha,
                    "output": None,
                    "status": "quarantined",
                    "reason": type(exc).__name__,
                }
            )

    planner = plan_fn or _resolve_planner()
    plans = planner(all_units, policy, secret)

    stats = DetectionStats()
    records_out: list[dict] = []
    for item, ctx, entry_idx in ingested:
        rendered = render_sanitized_file(item, ctx, plans, records_out, stats)
        dest = san / item.clean_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(rendered, encoding="utf-8", newline="")
        file_entries[entry_idx].update(output=f"sanitized/{item.clean_rel}", status="sanitized")

    with open(out / "records.jsonl", "w", encoding="utf-8") as f:
        for r in records_out:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

    n_san = sum(1 for e in file_entries if e.get("status") == "sanitized")
    n_gen = sum(1 for r in records_out if r.get("status") == "generalized")
    n_sup = sum(1 for r in records_out if r.get("status") == "suppressed")
    report = {
        "schema": "raksha-report/1",
        "files": file_entries,
        "counts": {
            "files_total": len(file_entries),
            "files_sanitized": n_san,
            "files_quarantined": len(file_entries) - n_san,
            "records_processed": len(records_out),
            "identifiers_detected": stats.identifiers,
            "records_linked": sum(1 for r in records_out if r.get("patient") is not None),
            "records_generalized": max(n_gen, stats.generalized),
            "records_suppressed": n_sup,
            "records_quarantined": 0,
        },
        "entities": dict(sorted(stats.by_type.items())),
        "timing": {"seconds": round(time.monotonic() - t0, 2)},
        "resources": {"peak_rss_mb": peak_rss_mb()},
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Enma / Secret Wards privacy firewall")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    policy = PolicyConfig.from_yaml(HERE / "config" / "policy.yaml")
    secret = os.environ.get("RAKSHA_SECRET", "").encode("utf-8")
    report = run_pipeline(Path(args.input), Path(args.output), policy, secret)
    c = report["counts"]
    # Strict side-channel rule: only aggregate counts on stdout
    print(
        f"files {c['files_total']} sanitized {c['files_sanitized']} "
        f"quarantined {c['files_quarantined']} records {c['records_processed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
