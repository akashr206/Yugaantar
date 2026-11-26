# Sync 2 Integration Guide (`main.py` Wiring)

Person 2's full pipeline is wrapped behind a single function in `rakshak/pipeline_p2.py`, plus `rakshak/security.py` helpers for quarantine and manifest export.

## 1. Ingestion → Person 2 (`RecordUnit`s)
Person 1's `ingest.py` + `schema_profiler.py` walks `sorted(input.rglob("*"))`, quarantines unsupported/corrupt files via `EphemeralVault.quarantine_file(...)`, and emits `list[RecordUnit]` (per `rakshak/contracts.py`).

## 2. Run Person 2 Pipeline (3 lines)
```python
from rakshak.contracts import PolicyConfig
from rakshak.pipeline_p2 import run_person2_pipeline
from rakshak.privacy import (
    apply_privacy_to_fhir_bundle,
    apply_privacy_to_structured_record,
    apply_privacy_to_text,
)
from rakshak.security import EphemeralVault, log_summary, safe_error_reason, sha256_hex, write_release_manifests

policy = PolicyConfig.from_yaml(HERE / "config" / "policy.yaml")
plans, date_orders = run_person2_pipeline(all_units, policy, SECRET)
```

## 3. Apply `PrivacyPlan` + Person 1 `detector.py` → `formats.py`
For each exported unit (`plan = plans[unit.unit_id]`):
1. Apply quasi-identifier transforms (`apply_privacy_to_structured_record`, `apply_privacy_to_fhir_bundle`, or `apply_privacy_to_text`).
2. Run Person 1's `detector.py` (names, relatives, phones, Aadhaar, ABHA, addresses, canaries, metadata, base64, headers, JSON keys, filename renaming).
3. Inject `plan.pseudonym` (`PSN-XXXXXX`) **inside** the exported record (when `not None`) and append `{"file": out_rel, "loc": loc, "patient": plan.pseudonym, ("status": "generalized")}` to `records_out`.
   - Note: Under Option C, `plan.generalization_level == "suppressed"` means **quasi-identifier suppression** (`keep_district=False`, `keep_pin_prefix=False`, widened `age_band`), keeping the record at its original `loc` and emitting `"status": "generalized"` in `records.jsonl`.

## 4. Write `records.jsonl` & `report.json` + Counter-Only Log
```python
report = write_release_manifests(
    out_dir=out,
    file_entries=file_entries,
    records_out=records_out,
    identifiers_detected=stats.identifiers,
    records_generalized=stats.generalized,
    records_suppressed=stats.suppressed,
    records_quarantined=0,
    elapsed_seconds=time.monotonic() - t0,
    secret=SECRET,
)
log_summary(report)
```
