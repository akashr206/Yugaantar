#!/usr/bin/env python3
"""Person 2 standalone development harness (mirrors starter/main.py contract).

Lets Person 2 build, run, and benchmark clue extraction, linkage, quasi-identifier
privacy, and security independently against validate_output.py and check_dev.py
without modifying Person 1's main.py until Sync 2.

Usage:
    python dev_p2_main.py --input ../dev/input --output ../out_p2
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import io
import json
import os
import re
import sys
import time
from pathlib import Path

from rakshak.clue_extractor import extract_all_clues, infer_file_date_orders
from rakshak.contracts import PolicyConfig, PrivacyPlan, RecordUnit
from rakshak.linkage import link_records
from rakshak.privacy import (
    apply_privacy_to_fhir_bundle,
    apply_privacy_to_structured_record,
    apply_privacy_to_text,
    build_privacy_plans,
)
from rakshak.security import (
    EphemeralVault,
    log_summary,
    safe_error_reason,
    sha256_hex,
    write_release_manifests,
)

HERE = Path(__file__).resolve().parent
POLICY = PolicyConfig.from_yaml(HERE / "config" / "policy.yaml")
SECRET = os.environ.get("RAKSHA_SECRET", "").encode("utf-8")

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford base32: no I, L, O, U

# Baseline regexes (kept only as fallback until Person 1 detector integrates)
BASELINE_PATTERNS = [
    ("PHONE", re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")),
    ("AADHAAR", re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")),
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("NAME", re.compile(r"(?i)(?<=name:)\s*[^\n,;|]+")),
]


class Stats:
    def __init__(self) -> None:
        self.identifiers = 0
        self.records = 0
        self.linked = 0
        self.generalized = 0
        self.suppressed = 0


def make_pseudonym(cluster_key: str, secret: bytes) -> str:
    """Stable, non-reversible HMAC-SHA256 pseudonym keyed by RAKSHA_SECRET."""
    digest = hmac.new(secret, cluster_key.encode("utf-8"), hashlib.sha256).digest()
    n = int.from_bytes(digest[:5], "big")
    code = "".join(_ALPHABET[(n >> (5 * i)) & 31] for i in range(6))
    return f"PSN-{code}"


def make_date_shift_days(key: str, secret: bytes, max_days: int) -> int:
    """Deterministic non-zero per-patient date shift in [-max_days, +max_days]."""
    digest = hmac.new(secret, b"shift|" + key.encode("utf-8"), hashlib.sha256).digest()
    val = int.from_bytes(digest[:4], "big") % (2 * max_days) - max_days
    return val if val != 0 else 1


def decode_bytes(raw: bytes) -> str | None:
    if raw.startswith(b"\xef\xbb\xbf"):
        try:
            return raw[3:].decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return None
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return raw.decode("utf-16")
        except UnicodeDecodeError:
            return None
    for enc in ("utf-8", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return None


def infer_hospital_hint(rel: str) -> str:
    low = rel.lower()
    if "sanjeevani" in low or "snj" in low:
        return "snj"
    if "jeevanjyoti" in low or "jjh" in low:
        return "jjh"
    if "arogya" in low:
        return "ass"
    if "dhhardoi" in low or "dhh" in low:
        return "dhh"
    return ""


def baseline_clean_text(text: str, stats: Stats) -> str:
    for kind, rx in BASELINE_PATTERNS:
        text, n = rx.subn(f"[{kind}]" if kind != "NAME" else " [NAME]", text)
        stats.identifiers += n
    return text


def baseline_clean_value(v, stats: Stats):
    if isinstance(v, str):
        return baseline_clean_text(v, stats)
    if isinstance(v, list):
        return [baseline_clean_value(x, stats) for x in v]
    if isinstance(v, dict):
        return {baseline_clean_text(k, stats): baseline_clean_value(x, stats) for k, x in v.items()}
    return v


def _is_doc_json(data: object) -> bool:
    if not isinstance(data, dict):
        return True
    if data.get("resourceType") == "Bundle":
        return True
    # Single-patient diagnostic reports (have top-level "patient" dict or "panels") or helpdesk/index docs
    if isinstance(data.get("patient"), dict) or "panels" in data or "tickets" in data:
        return True
    return not any(
        isinstance(v, list) and v and all(isinstance(x, dict) for x in v)
        for v in data.values()
    )


def parse_file_units(rel: str, ext: str, text: str) -> tuple[str, object, list[RecordUnit]]:
    """Parse one decoded file into RecordUnits for Person 2's pipeline."""
    hosp = infer_hospital_hint(rel)
    units: list[RecordUnit] = []
    if ext in (".csv", ".tsv"):
        delim = "\t" if ext == ".tsv" else ","
        rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=delim))
        if not rows:
            return "csv", (delim, [], []), units
        header, body = rows[0], rows[1:]
        for i, row in enumerate(body):
            loc = f"row:{i}"
            fields = {header[j]: row[j] for j in range(min(len(header), len(row)))}
            units.append(
                RecordUnit(
                    unit_id=f"{rel}::{loc}",
                    file_rel=rel,
                    loc=loc,
                    format_type="tsv" if ext == ".tsv" else "csv",
                    hospital_hint=hosp,
                    raw_text=delim.join(row),
                    structured_fields=fields,
                    metadata_hints={"filename": Path(rel).name},
                )
            )
        return "csv", (delim, header, body), units

    if ext == ".json":
        data = json.loads(text)
        is_fhir = isinstance(data, dict) and data.get("resourceType") == "Bundle"
        is_doc = _is_doc_json(data)
        if is_doc:
            loc = "doc"
            units.append(
                RecordUnit(
                    unit_id=f"{rel}::{loc}",
                    file_rel=rel,
                    loc=loc,
                    format_type="fhir" if is_fhir else "json",
                    hospital_hint=hosp,
                    raw_text=text,
                    structured_fields=data if isinstance(data, dict) else {"value": data},
                    metadata_hints={"filename": Path(rel).name},
                )
            )
        else:
            meta = data.get("_export_meta") if isinstance(data.get("_export_meta"), dict) else {}
            meta_hints = {"filename": Path(rel).name}
            for mk, mv in meta.items():
                if isinstance(mv, str):
                    meta_hints[f"_export_meta.{mk}"] = mv
            for k, v in data.items():
                if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                    pointer_key = k.replace("~", "~0").replace("/", "~1")
                    for i, item in enumerate(v):
                        loc = f"/{pointer_key}/{i}"
                        units.append(
                            RecordUnit(
                                unit_id=f"{rel}::{loc}",
                                file_rel=rel,
                                loc=loc,
                                format_type="json",
                                hospital_hint=hosp,
                                raw_text=json.dumps(item, ensure_ascii=False),
                                structured_fields=item,
                                metadata_hints=meta_hints,
                            )
                        )
        return "json", data, units

    loc = "doc"
    units.append(
        RecordUnit(
            unit_id=f"{rel}::{loc}",
            file_rel=rel,
            loc=loc,
            format_type="txt",
            hospital_hint=hosp,
            raw_text=text,
            metadata_hints={"filename": Path(rel).name},
        )
    )
    return "txt", text, units


_ID_KEY_RE = re.compile(r"^(?:uhid|mrn|reg[_\s]*no|patient[_\s]*ref|patientid|patient_id|hospital_uhid)$", re.I)


def _inject_pseudonym_json_obj(obj: dict, pid: str) -> None:
    for k in list(obj.keys()):
        if _ID_KEY_RE.match(k):
            obj[k] = pid
            return
    for v in obj.values():
        if isinstance(v, dict):
            for k2 in list(v.keys()):
                if _ID_KEY_RE.match(k2):
                    v[k2] = pid
                    return
    obj["patient_ref"] = pid


def render_sanitized(
    kind: str,
    parsed: object,
    rel: str,
    plans: dict[str, PrivacyPlan],
    date_orders: dict[tuple[str, str], str],
    records_out: list[dict],
    stats: Stats,
) -> str:
    is_clean_doc = rel.startswith("clean/") or rel == "misc/README_export.txt"
    if kind == "csv":
        delim, header, body = parsed  # type: ignore[misc]
        buf = io.StringIO()
        w = csv.writer(buf, delimiter=delim, lineterminator="\n")
        if header or body:
            w.writerow(header)
        id_col = next((idx for idx, h in enumerate(header) if _ID_KEY_RE.match(h.strip())), None)
        for i, row in enumerate(body):
            loc = f"row:{i}"
            plan = plans[f"{rel}::{loc}"]
            gen_changed = False
            if not is_clean_doc and header:
                row_dict = {header[j]: row[j] for j in range(min(len(header), len(row)))}
                t_dict, gen_changed = apply_privacy_to_structured_record(row_dict, plan, rel, date_orders, POLICY)
                priv_row = [str(t_dict.get(header[j], row[j])) if j < len(header) else row[j] for j in range(len(row))]
            else:
                priv_row = list(row)
            cleaned_row = [baseline_clean_text(c, stats) for c in priv_row]
            if plan.pseudonym is not None:
                if id_col is not None and id_col < len(cleaned_row):
                    cleaned_row[id_col] = plan.pseudonym
                elif cleaned_row:
                    cleaned_row[0] = f"{plan.pseudonym}"
            w.writerow(cleaned_row)
            rec_entry: dict = {"file": f"sanitized/{rel}", "loc": loc, "patient": plan.pseudonym}
            if gen_changed:
                rec_entry["status"] = "generalized"
                stats.generalized += 1
            records_out.append(rec_entry)
            stats.records += 1
            if plan.pseudonym is not None:
                stats.linked += 1
        return buf.getvalue()

    if kind == "json":
        is_fhir = isinstance(parsed, dict) and parsed.get("resourceType") == "Bundle"
        is_doc = _is_doc_json(parsed)
        if is_doc:
            plan = plans[f"{rel}::doc"]
            gen_changed = False
            working = parsed
            if not is_clean_doc and isinstance(working, dict):
                if is_fhir:
                    import copy
                    working, gen_changed = apply_privacy_to_fhir_bundle(copy.deepcopy(working), plan, POLICY)
                else:
                    working, gen_changed = apply_privacy_to_structured_record(working, plan, rel, date_orders, POLICY)
            data = baseline_clean_value(working, stats)
            if plan.pseudonym is not None and isinstance(data, dict):
                if is_fhir:
                    for entry in data.get("entry") or []:
                        res = entry.get("resource") if isinstance(entry, dict) else None
                        if isinstance(res, dict) and res.get("resourceType") == "Patient":
                            res["identifier"] = [{"system": "urn:raksha:pseudonym", "value": plan.pseudonym}]
                else:
                    _inject_pseudonym_json_obj(data, plan.pseudonym)
            rec_entry = {"file": f"sanitized/{rel}", "loc": "doc", "patient": plan.pseudonym}
            if gen_changed:
                rec_entry["status"] = "generalized"
                stats.generalized += 1
            records_out.append(rec_entry)
            stats.records += 1
            if plan.pseudonym is not None:
                stats.linked += 1
            return json.dumps(data, ensure_ascii=False, indent=2) + "\n"

        # Multi-record JSON (opd, bulk_lab, dhh_patients)
        import copy
        working_root = copy.deepcopy(parsed)
        if isinstance(working_root, dict):
            for k, v in working_root.items():
                if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                    pointer_key = k.replace("~", "~0").replace("/", "~1")
                    for i, item in enumerate(v):
                        loc = f"/{pointer_key}/{i}"
                        plan = plans[f"{rel}::{loc}"]
                        gen_changed = False
                        if not is_clean_doc:
                            item, gen_changed = apply_privacy_to_structured_record(item, plan, rel, date_orders, POLICY)
                            v[i] = item
                        if plan.pseudonym is not None and isinstance(item, dict):
                            _inject_pseudonym_json_obj(item, plan.pseudonym)
                        rec_entry = {"file": f"sanitized/{rel}", "loc": loc, "patient": plan.pseudonym}
                        if gen_changed:
                            rec_entry["status"] = "generalized"
                            stats.generalized += 1
                        records_out.append(rec_entry)
                        stats.records += 1
                        if plan.pseudonym is not None:
                            stats.linked += 1
        data = baseline_clean_value(working_root, stats)
        return json.dumps(data, ensure_ascii=False, indent=2) + "\n"

    plan = plans[f"{rel}::doc"]
    txt = str(parsed)
    gen_changed = False
    if not is_clean_doc:
        txt, gen_changed = apply_privacy_to_text(txt, plan, date_orders.get((rel, "__doc__"), "dmy"))
    rec_entry = {"file": f"sanitized/{rel}", "loc": "doc", "patient": plan.pseudonym}
    if gen_changed:
        rec_entry["status"] = "generalized"
        stats.generalized += 1
    records_out.append(rec_entry)
    stats.records += 1
    if plan.pseudonym is not None:
        stats.linked += 1
    cleaned_txt = baseline_clean_text(txt, stats)
    if plan.pseudonym is not None:
        return f"Patient: {plan.pseudonym}\n{cleaned_txt}"
    return cleaned_txt


def peak_rss_mb() -> float:
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(r / 1024 if sys.platform != "darwin" else r / 1048576, 1)
    except Exception:
        return 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Person 2 standalone test harness")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    t0 = time.monotonic()
    inp, out = Path(args.input), Path(args.output)
    san = out / "sanitized"
    quar_dir = out / "quarantine"
    san.mkdir(parents=True, exist_ok=True)
    sys.setrecursionlimit(5000)

    vault = EphemeralVault()
    stats = Stats()
    file_entries: list[dict] = []
    parsed_files: list[tuple[str, str, object, int]] = []
    all_units: list[RecordUnit] = []

    for path in sorted(p for p in inp.rglob("*") if p.is_file()):
        rel = path.relative_to(inp).as_posix()
        raw = path.read_bytes()
        in_sha = sha256_hex(raw)
        entry: dict = {"input_sha256": in_sha}
        ext = path.suffix.lower()
        try:
            if ext not in (".csv", ".tsv", ".json", ".txt"):
                raise ValueError("unsupported_format")
            if not raw.strip():
                raise ValueError("empty_file")
            text = decode_bytes(raw)
            if text is None:
                raise ValueError("undecodable_text")
            kind, parsed, units = parse_file_units(rel, ext, text)
            idx = len(file_entries)
            file_entries.append(entry)
            parsed_files.append((rel, kind, parsed, idx))
            all_units.extend(units)
        except (ValueError, RecursionError, UnicodeError, csv.Error) as exc:
            reason = safe_error_reason(exc)
            vault.quarantine_file(quar_dir, in_sha, raw)
            entry.update(output=None, status="quarantined", reason=reason)
            file_entries.append(entry)

    date_orders = infer_file_date_orders(all_units)
    clues = extract_all_clues(all_units)
    cluster_map = link_records(clues)
    plans = build_privacy_plans(clues, cluster_map, POLICY, SECRET)

    records_out: list[dict] = []
    for rel, kind, parsed, entry_idx in parsed_files:
        rendered = render_sanitized(kind, parsed, rel, plans, date_orders, records_out, stats)
        dest = san / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(rendered, encoding="utf-8", newline="")
        file_entries[entry_idx].update(output=f"sanitized/{rel}", status="sanitized")

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
