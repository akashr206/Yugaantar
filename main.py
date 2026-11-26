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
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

from rakshak.clue_extractor import extract_clues_for_unit, infer_file_date_orders
from rakshak.contracts import IdentityClues, PolicyConfig, PrivacyPlan, RecordUnit
from rakshak.detector import DetectionStats
from rakshak.formats import build_record_units, render_sanitized_file
from rakshak.ingest import (
    IngestResult,
    QuarantineError,
    compute_corpus_common_tokens,
    ingest_file,
    sanitize_rel_path,
)
from rakshak.planner import plan_privacy_full
from rakshak.security import (
    EphemeralVault,
    log_summary,
    safe_error_reason,
    sha256_hex,
    write_release_manifests,
)
from rakshak.translit import HONORIFIC_TOKENS

HERE = Path(__file__).resolve().parent


def _build_path_pii_context(
    clues: list[IdentityClues],
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, bool], frozenset[str], tuple[frozenset[str], ...]]:
    """Build per-file and corpus-wide PII token/ID lookups from Pass 1 clues."""
    file_tokens: dict[str, set[str]] = defaultdict(set)
    file_ids: dict[str, set[str]] = defaultdict(set)
    file_has_pii: dict[str, bool] = defaultdict(bool)
    corpus_ids: set[str] = set()
    name_pairs_set: set[frozenset[str]] = set()

    for cl in clues:
        rel = cl.file_rel
        has_any = bool(
            cl.patient_names
            or cl.relative_names
            or cl.phones
            or cl.aadhaars
            or cl.abhas
            or cl.mrns
            or cl.emails
            or cl.pans
            or cl.insurance_ids
        )
        if has_any:
            file_has_pii[rel] = True

        for nc in (*cl.patient_names, *cl.relative_names):
            toks = [t for t in nc.latin_tokens if len(t) >= 3 and t not in HONORIFIC_TOKENS]
            for t in toks:
                file_tokens[rel].add(t)
            for pt in nc.phonetic_tokens:
                if len(pt) >= 3:
                    file_tokens[rel].add(pt)
            if len(toks) >= 2:
                name_pairs_set.add(frozenset(toks))

        for addr in (*cl.emails, *cl.abha_addresses):
            local = addr.split("@", 1)[0]
            for p in re.split(r"[^a-z0-9]+", local.lower()):
                if len(p) >= 3 and not p.isdigit():
                    file_tokens[rel].add(p)

        for raw_id in (*cl.phones, *cl.aadhaars, *cl.abhas, *cl.mrns, *cl.pans, *cl.insurance_ids):
            compact = re.sub(r"[^a-z0-9]", "", raw_id.lower())
            if len(compact) >= 5:
                file_ids[rel].add(compact)
                corpus_ids.add(compact)

    return file_tokens, file_ids, file_has_pii, frozenset(corpus_ids), tuple(name_pairs_set)


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

    for path in sorted(p for p in inp.rglob("*") if p.is_file()):
        try:
            item = ingest_file(path, inp, secret)
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

    if plan_fn is None:
        plans, date_orders, clues = plan_privacy_full(all_units, policy, secret)
    else:
        plans = plan_fn(all_units, policy, secret)
        date_orders = infer_file_date_orders(all_units)
        clues = [extract_clues_for_unit(u, date_orders) for u in all_units]

    corpus_common = compute_corpus_common_tokens([it.file_rel for it, _, _ in ingested])
    file_tokens, file_ids, file_has_pii, corpus_ids, corpus_name_pairs = _build_path_pii_context(clues)

    seen_out_paths: set[str] = set()
    resolved_ingested: list[tuple[IngestResult, dict, int]] = []
    for item, ctx, entry_idx in ingested:
        rel = item.file_rel
        clean_doc = bool(ctx.get("clean_doc", False))
        # Use file-specific IDs/tokens + corpus-wide IDs and full-name pairs (skip single-token corpus match to avoid noise)
        combined_ids = frozenset(file_ids.get(rel, set()) | corpus_ids)
        clean_rel, renamed = sanitize_rel_path(
            rel,
            secret,
            known_pii_tokens=frozenset(file_tokens.get(rel, set())),
            known_pii_ids=combined_ids,
            corpus_common_tokens=corpus_common,
            corpus_name_pairs=corpus_name_pairs,
            file_has_pii=(not clean_doc) and file_has_pii.get(rel, False),
        )
        if clean_rel in seen_out_paths:
            stem = Path(clean_rel).stem
            ext = Path(clean_rel).suffix
            parent = Path(clean_rel).parent.as_posix()
            suffix_hash = hmac.new(secret, rel.encode("utf-8"), hashlib.sha256).hexdigest()[:6]
            clean_rel = f"{parent}/{stem}_{suffix_hash}{ext}" if parent != "." else f"{stem}_{suffix_hash}{ext}"
            renamed = True
        seen_out_paths.add(clean_rel)
        if clean_rel != item.clean_rel or renamed != item.filename_renamed:
            item = IngestResult(
                file_rel=item.file_rel,
                sha256=item.sha256,
                ext=item.ext,
                format_type=item.format_type,
                text=item.text,
                parsed=item.parsed,
                hospital_hint=item.hospital_hint,
                clean_rel=clean_rel,
                filename_renamed=renamed,
            )
        resolved_ingested.append((item, ctx, entry_idx))

    stats = DetectionStats()
    records_out: list[dict] = []
    for item, ctx, entry_idx in resolved_ingested:
        rendered = render_sanitized_file(item, ctx, plans, records_out, stats, date_orders=date_orders)
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
