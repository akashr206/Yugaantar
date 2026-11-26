"""Format-preserving & FHIR R4 sanitized writers and RecordUnit builders."""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import re
from functools import lru_cache
from typing import Any

from rakshak.contracts import DateMention, PrivacyPlan, RecordUnit
from rakshak.detector import (
    DetectionStats,
    clean_free_text,
    format_age_band,
    generalize_ages_in_text,
    sanitize_structured_cell,
    shift_dates_in_text,
)
from rakshak.ingest import IngestResult
from rakshak.schema_profiler import (
    ROLE_AADHAAR,
    ROLE_ABHA,
    ROLE_ADMISSION_DATE,
    ROLE_AGE,
    ROLE_AGE_SEX,
    ROLE_DATE,
    ROLE_DISCHARGE_DATE,
    ROLE_DISTRICT,
    ROLE_DOB,
    ROLE_EMAIL,
    ROLE_MRN,
    ROLE_PATIENT_NAME,
    ROLE_PHONE,
    ROLE_PIN,
    ROLE_PROTECTED_CLINICAL,
    ROLE_RELATIVE_NAME,
    ROLE_UNKNOWN,
    classify_column,
    encode_base64_text,
    extract_dates_from_text,
    is_clean_reference_file,
    parse_date_string,
    profile_table,
    try_decode_base64_text,
)

_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_RECORD_ID_KEYS = ("uhid", "patient_ref", "hospital_uhid", "patientid", "patient_id", "reg_no")


def _record_status(plan: PrivacyPlan) -> str:
    if plan.generalization_level in ("widened", "suppressed") or plan.age_band is not None or not plan.keep_district or not plan.keep_pin_prefix:
        return "generalized"
    return "sanitized"


def _extract_csv_dates(row: list[str], profiles: list, loc: str) -> tuple[DateMention, ...]:
    mentions: list[DateMention] = []
    for idx, prof in enumerate(profiles):
        if idx >= len(row):
            continue
        val = row[idx].strip()
        if not val:
            continue
        if prof.role in (ROLE_DOB, ROLE_DATE, ROLE_ADMISSION_DATE, ROLE_DISCHARGE_DATE):
            parsed = parse_date_string(val, preferred_order=prof.date_order)
            role_map = {
                ROLE_DOB: "dob",
                ROLE_ADMISSION_DATE: "admission",
                ROLE_DISCHARGE_DATE: "discharge",
                ROLE_DATE: "clinical",
            }
            mentions.append(
                DateMention(
                    raw_text=val,
                    iso_date=parsed.isoformat() if parsed else None,
                    role=role_map[prof.role],
                    date_order=prof.date_order,
                    field_path=f"col:{prof.raw_name}",
                )
            )
    return tuple(mentions)


def _extract_json_dates_from_obj(obj: dict[str, Any], hospital_hint: str) -> tuple[DateMention, ...]:
    mentions: list[DateMention] = []
    stack: list[tuple[str, Any]] = [("", obj)]
    while stack:
        prefix, cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                path = f"{prefix}/{k}"
                if isinstance(v, str) and v.strip():
                    low = k.lower()
                    if any(t in low for t in ("dob", "birthdate", "date", "collected", "timestamp", "recordeddate", "authoredon", "start", "end")):
                        parsed = parse_date_string(v, preferred_order="iso" if "-" in v[:5] else ("mdy" if hospital_hint == "ass" else "dmy"))
                        if parsed:
                            role = "dob" if ("dob" in low or "birth" in low) else (
                                "admission" if low == "start" else ("discharge" if low == "end" else "clinical")
                            )
                            mentions.append(
                                DateMention(
                                    raw_text=v,
                                    iso_date=parsed.isoformat(),
                                    role=role,
                                    date_order="iso" if "-" in v[:5] else "dmy",
                                    field_path=path,
                                )
                            )
                elif isinstance(v, (dict, list)):
                    stack.append((path, v))
        elif isinstance(cur, list):
            for idx, item in enumerate(cur):
                if isinstance(item, (dict, list)):
                    stack.append((f"{prefix}/{idx}", item))
    return tuple(mentions)


def _is_doc_json_container(data: Any) -> bool:
    if not isinstance(data, dict):
        return True
    if data.get("resourceType") == "Bundle":
        return True
    # Single-patient diagnostic reports (have top-level "patient" dict or "panels") or helpdesk docs
    if isinstance(data.get("patient"), dict) or "panels" in data or "tickets" in data:
        return True
    return not any(isinstance(v, list) and v and all(isinstance(x, dict) for x in v) for v in data.values())


def build_record_units(item: IngestResult) -> tuple[list[RecordUnit], dict[str, Any]]:
    """Build RecordUnit objects and file profiling context from an IngestResult."""
    rel = item.file_rel
    clean_doc = is_clean_reference_file(rel, item.format_type, item.parsed, item.text)
    units: list[RecordUnit] = []
    ctx: dict[str, Any] = {"clean_doc": clean_doc}

    if item.format_type in ("csv", "tsv"):
        delim, header, body = item.parsed
        profiles = profile_table(header, body, hospital_hint=item.hospital_hint)
        ctx["profiles"] = profiles
        for i, row in enumerate(body):
            loc = f"row:{i}"
            fields = {header[j]: row[j] for j in range(min(len(header), len(row)))}
            units.append(
                RecordUnit(
                    unit_id=f"{rel}::{loc}",
                    file_rel=rel,
                    loc=loc,
                    format_type=item.format_type,
                    hospital_hint=item.hospital_hint,
                    raw_text=delim.join(row),
                    structured_fields=fields,
                    extracted_dates=() if clean_doc else _extract_csv_dates(row, profiles, loc),
                    metadata_hints={"filename": rel},
                )
            )
        return units, ctx

    if item.format_type in ("json", "fhir"):
        data = item.parsed
        is_doc = _is_doc_json_container(data)
        ctx["is_doc"] = is_doc
        if is_doc:
            loc = "doc"
            units.append(
                RecordUnit(
                    unit_id=f"{rel}::{loc}",
                    file_rel=rel,
                    loc=loc,
                    format_type=item.format_type,
                    hospital_hint=item.hospital_hint,
                    raw_text=item.text,
                    structured_fields=data if isinstance(data, dict) else {"value": data},
                    extracted_dates=() if clean_doc or not isinstance(data, dict) else _extract_json_dates_from_obj(data, item.hospital_hint),
                    metadata_hints={"filename": rel},
                )
            )
        else:
            meta = data.get("_export_meta") if isinstance(data.get("_export_meta"), dict) else {}
            meta_hints = {"filename": rel}
            for mk, mv in meta.items():
                if isinstance(mv, str):
                    meta_hints[f"_export_meta.{mk}"] = mv
            for k, v in data.items():
                if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                    pointer_key = k.replace("~", "~0").replace("/", "~1")
                    for i, rec in enumerate(v):
                        loc = f"/{pointer_key}/{i}"
                        units.append(
                            RecordUnit(
                                unit_id=f"{rel}::{loc}",
                                file_rel=rel,
                                loc=loc,
                                format_type="json",
                                hospital_hint=item.hospital_hint,
                                raw_text=json.dumps(rec, ensure_ascii=False),
                                structured_fields=rec,
                                extracted_dates=() if clean_doc else _extract_json_dates_from_obj(rec, item.hospital_hint),
                                metadata_hints=meta_hints,
                            )
                        )
        return units, ctx

    loc = "doc"
    units.append(
        RecordUnit(
            unit_id=f"{rel}::{loc}",
            file_rel=rel,
            loc=loc,
            format_type="txt",
            hospital_hint=item.hospital_hint,
            raw_text=item.text,
            extracted_dates=() if clean_doc else extract_dates_from_text(item.text, hospital_hint=item.hospital_hint),
            metadata_hints={"filename": rel},
        )
    )
    return units, ctx


def _sanitize_json_scalar(
    key: str,
    val: Any,
    plan: PrivacyPlan | None,
    stats: DetectionStats,
    hospital_hint: str,
    clean_doc: bool,
    inside_practitioner: bool = False,
    parent_key: str = "",
) -> Any:
    """Sanitize a scalar JSON leaf while keeping value types valid where required."""
    if clean_doc:
        if isinstance(val, str):
            return clean_free_text(val, stats, clean_only_direct_ids=True, hospital_hint=hospital_hint)
        return val

    if inside_practitioner:
        return val

    # Clinical sub-object names (e.g. panels[].name = "Routine", hospital/facility.name, Organization.name)
    low_key = key.lower()
    low_parent = parent_key.lower()
    if low_key == "name" and low_parent in ("panels", "facility", "hospital", "organization", "lab", "wards"):
        return val

    prof = classify_column(key, [str(val)] if val is not None else [], hospital_hint=hospital_hint)
    is_suppressed = plan is not None and plan.generalization_level == "suppressed"

    # Handle numeric PII/quasi-identifiers (e.g. integer Aadhaar in dhh_patients.json or integer age)
    if isinstance(val, int) and not isinstance(val, bool):
        if prof.role == ROLE_AADHAAR or len(str(abs(val))) == 12:
            stats.add("AADHAAR")
            return "[AADHAAR]"
        if prof.role == ROLE_PHONE or (len(str(abs(val))) == 10 and str(abs(val))[0] in "6789"):
            stats.add("PHONE")
            return "[PHONE]"
        if prof.role == ROLE_PIN or (len(str(abs(val))) == 6 and low_key in ("pin", "pincode", "zip", "postalcode")):
            stats.add("PIN")
            if plan and not plan.keep_pin_prefix:
                return "[PIN]"
            stats.generalized += 1
            return f"{str(val)[:3]}***"
        if prof.role == ROLE_AGE:
            stats.generalized += 1
            if plan and plan.age_band:
                return plan.age_band
            if is_suppressed:
                return "[AGE]"
            return format_age_band(val)
        return val

    if not isinstance(val, str):
        return val

    # Semantic overrides for nested JSON person/guardian sub-fields
    if low_parent == "name" and low_key == "local":
        return sanitize_structured_cell(val, ROLE_PATIENT_NAME, "dmy", plan, stats, hospital_hint=hospital_hint)
    if low_parent == "guardian" and low_key == "name":
        return sanitize_structured_cell(val, ROLE_RELATIVE_NAME, "dmy", plan, stats, hospital_hint=hospital_hint)

    # Protect structural UUIDs/URNs/HTTP ontology URLs from ID-regex corruption
    if low_key in ("reference", "fullurl", "system", "url", "code", "version", "id", "profile") and re.fullmatch(
        r"(?:urn:uuid:[0-9a-fA-F-]{36}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}|https?://\S+)",
        val.strip(),
    ):
        return val

    # Base64 attachments (e.g. scanned_consent_form.json, FHIR DocumentReference)
    decoded_b64 = try_decode_base64_text(val)
    if decoded_b64 is not None:
        cleaned_inner = clean_free_text(decoded_b64, stats, plan=plan, hospital_hint=hospital_hint)
        return encode_base64_text(cleaned_inner)

    # _export_meta.source_path special scrub (e.g. D:\HIS\exports\opd\Anita_Sharma_010565\visits_2026_02.json)
    if low_key in ("source_path", "export_path", "file_path"):
        stats.add("PATIENT_NAME")
        return re.sub(r"([\\/])[^\\/]+([\\/][^\\/]+\.json)$", r"\1[REDACTED]\2", val)

    # JSON keys that hold patient identifiers directly
    if prof.role == ROLE_MRN and low_key in _RECORD_ID_KEYS:
        stats.add("MRN")
        return plan.pseudonym if (plan and plan.pseudonym) else "[MRN]"

    if prof.role == ROLE_PROTECTED_CLINICAL:
        return clean_free_text(val, stats, plan=None, clean_only_direct_ids=True, hospital_hint=hospital_hint)

    if prof.role != ROLE_UNKNOWN:
        return sanitize_structured_cell(val, prof.role, prof.date_order, plan, stats, hospital_hint=hospital_hint)

    return clean_free_text(val, stats, plan=plan, hospital_hint=hospital_hint)


@lru_cache(maxsize=4096)
def _clean_json_key_cached(k: str, hospital_hint: str) -> tuple[str, tuple[tuple[str, int], ...]]:
    tmp_stats = DetectionStats()
    c = clean_free_text(k, tmp_stats, plan=None, hospital_hint=hospital_hint)
    if c == k and re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}", k.strip()):
        tmp_stats.add("PATIENT_NAME")
        c = "[PATIENT_NAME]"
    return c, tuple(tmp_stats.by_type.items())


def _sanitize_json_iterative(
    root: Any,
    plan: PrivacyPlan | None,
    stats: DetectionStats,
    hospital_hint: str,
    clean_doc: bool,
) -> Any:
    """Iteratively sanitize a JSON container without Python recursion limits."""
    def _clean_key(k: str) -> str:
        if clean_doc:
            return k
        c, stat_items = _clean_json_key_cached(k, hospital_hint)
        for kind, cnt in stat_items:
            stats.add(kind, cnt)
        return c

    if not isinstance(root, (dict, list)):
        return _sanitize_json_scalar("", root, plan, stats, hospital_hint, clean_doc)

    new_root: Any = {} if isinstance(root, dict) else []
    # Stack items: (src_node, dst_node, parent_key, inside_protected_resource)
    stack: list[tuple[Any, Any, str, bool]] = [(root, new_root, "", False)]

    while stack:
        src, dst, pkey, in_prac = stack.pop()
        if isinstance(src, dict):
            rtype = src.get("resourceType")
            is_prac_obj = in_prac or rtype in ("Practitioner", "Organization")
            for raw_k, v in src.items():
                ck = _clean_key(raw_k)
                # Ensure unique key if multiple PII keys collapse to "[PATIENT_NAME]" or "[MRN]"
                if ck != raw_k and ck in dst:
                    idx = 1
                    while f"{ck}_{idx}" in dst:
                        idx += 1
                    ck = f"{ck}_{idx}"
                if isinstance(v, dict):
                    child: dict[str, Any] = {}
                    dst[ck] = child
                    stack.append((v, child, raw_k, is_prac_obj))
                elif isinstance(v, list):
                    child_list: list[Any] = []
                    dst[ck] = child_list
                    stack.append((v, child_list, raw_k, is_prac_obj))
                else:
                    dst[ck] = _sanitize_json_scalar(
                        raw_k,
                        v,
                        plan,
                        stats,
                        hospital_hint,
                        clean_doc,
                        inside_practitioner=is_prac_obj,
                        parent_key=pkey,
                    )
        elif isinstance(src, list):
            for item in src:
                if isinstance(item, dict):
                    child_d: dict[str, Any] = {}
                    dst.append(child_d)
                    stack.append((item, child_d, pkey, in_prac))
                elif isinstance(item, list):
                    child_l: list[Any] = []
                    dst.append(child_l)
                    stack.append((item, child_l, pkey, in_prac))
                else:
                    dst.append(
                        _sanitize_json_scalar(
                            pkey,
                            item,
                            plan,
                            stats,
                            hospital_hint,
                            clean_doc,
                            inside_practitioner=in_prac,
                            parent_key=pkey,
                        )
                    )
    return new_root


def _sanitize_fhir_bundle(
    bundle: dict[str, Any],
    plan: PrivacyPlan,
    stats: DetectionStats,
    hospital_hint: str,
) -> dict[str, Any]:
    """Sanitize a FHIR R4 Document Bundle while keeping references valid and Patient PSN injected."""
    cleaned = _sanitize_json_iterative(bundle, plan, stats, hospital_hint, clean_doc=False)
    if not isinstance(cleaned, dict):
        return cleaned

    psn = plan.pseudonym
    # Scrub Bundle.meta.source (#SNJ/26/064545 -> #PSN-XXXXXX or #[MRN])
    meta = cleaned.get("meta")
    if isinstance(meta, dict) and isinstance(meta.get("source"), str):
        stats.add("MRN")
        meta["source"] = f"#{psn}" if psn else "#[MRN]"

    for ent in cleaned.get("entry") or []:
        if not isinstance(ent, dict):
            continue
        res = ent.get("resource")
        if not isinstance(res, dict):
            continue
        rtype = res.get("resourceType")
        if rtype == "Composition":
            subj = res.get("subject")
            if isinstance(subj, dict) and "display" in subj:
                stats.add("PATIENT_NAME")
                subj["display"] = psn if psn else "[PATIENT_NAME]"
        elif rtype == "Patient":
            # Replace identifiers with clean pseudonym identifier (keep FHIR identifier array valid)
            res["identifier"] = [
                {"system": "https://enma.raksha.local/pseudonym", "value": psn or "[MRN]"}
            ]
            # Replace Patient.name cleanly (catches HumanName text/given/family)
            stats.add("PATIENT_NAME")
            res["name"] = [{"text": psn or "[PATIENT_NAME]"}]
            # Scrub Patient.telecom explicitly so +91- phones/emails never survive
            if isinstance(res.get("telecom"), list):
                for tc in res["telecom"]:
                    if isinstance(tc, dict) and isinstance(tc.get("value"), str):
                        sys_low = str(tc.get("system", "")).lower()
                        kind = "EMAIL" if sys_low == "email" or "@" in tc["value"] else "PHONE"
                        stats.add(kind)
                        tc["value"] = f"[{kind}]"
            # Suppress Patient.birthDate by omitting the optional element on widened/suppressed tiers (keeps valid FHIR R4)
            if "birthDate" in res and (
                plan.generalization_level != "standard"
                or not re.fullmatch(r"\d{4}(?:-\d{2}(?:-\d{2})?)?", str(res["birthDate"]))
            ):
                res.pop("birthDate", None)
            # Sanitize narrative div cleanly
            txt = res.get("text")
            if isinstance(txt, dict) and isinstance(txt.get("div"), str):
                age_part = plan.age_band or "[AGE]"
                gender = res.get("gender") or ""
                label = psn or "[PATIENT_NAME]"
                txt["div"] = f'<div xmlns="http://www.w3.org/1999/xhtml">{label}, {age_part}, {gender}</div>'
            # Generalize Patient.address to district/PIN-prefix only
            addrs = res.get("address")
            if isinstance(addrs, list):
                for a in addrs:
                    if not isinstance(a, dict):
                        continue
                    a.pop("line", None)
                    a.pop("city", None)
                    a.pop("text", None)
                    if not plan.keep_district:
                        a["district"] = "[DISTRICT]"
                    if "postalCode" in a:
                        pc = str(a["postalCode"]).translate(_DEVA_DIGITS)
                        a["postalCode"] = f"{pc[:3]}***" if (plan.keep_pin_prefix and len(pc) >= 3) else "[PIN]"
            # Remove contact (relative phone/name) completely
            if "contact" in res:
                stats.add("RELATIVE_NAME")
                res["contact"] = [{"relationship": [{"text": "relative"}], "name": {"text": "[RELATIVE_NAME]"}}]
            # Sanitize extensions like age or mothers-name / relative-name
            for ext in res.get("extension") or []:
                if not isinstance(ext, dict):
                    continue
                url_low = str(ext.get("url", "")).lower()
                if "age" in url_low and "valueInteger" in ext:
                    val_int = ext.pop("valueInteger")
                    ext["valueString"] = plan.age_band or ("[AGE]" if plan.generalization_level == "suppressed" else format_age_band(int(val_int)))
                elif any(k in url_low for k in ("name", "mother", "father", "relative", "guardian")) and "valueString" in ext:
                    stats.add("RELATIVE_NAME")
                    ext["valueString"] = "[RELATIVE_NAME]"
        elif rtype == "Encounter":
            if "identifier" in res and isinstance(res["identifier"], list):
                for ident in res["identifier"]:
                    if isinstance(ident, dict) and "value" in ident:
                        stats.add("MRN")
                        ident["value"] = psn if psn else "[MRN]"
        elif rtype == "Organization":
            if not plan.keep_district and isinstance(res.get("address"), list):
                for a in res["address"]:
                    if isinstance(a, dict):
                        a.pop("city", None)
    return cleaned


def render_sanitized_file(
    item: IngestResult,
    ctx: dict[str, Any],
    plans: dict[str, PrivacyPlan],
    records_out: list[dict[str, Any]],
    stats: DetectionStats,
) -> str:
    """Render one sanitized file and append its records.jsonl entries."""
    rel = item.file_rel
    out_rel = item.clean_rel
    clean_doc: bool = bool(ctx.get("clean_doc", False))

    if item.format_type in ("csv", "tsv"):
        delim, header, body = item.parsed
        profiles = ctx["profiles"]
        buf = io.StringIO()
        writer = csv.writer(buf, delimiter=delim, lineterminator="\n")

        # Rename only headers that embed PII (e.g. ward_vitals_sheet.csv)
        out_header = []
        id_col_idx = None
        fallback_id_col_idx = None
        for idx, prof in enumerate(profiles):
            if prof.role == ROLE_MRN:
                low_h = re.sub(r"[^a-z0-9]+", "_", prof.raw_name.lower()).strip("_")
                if low_h in _RECORD_ID_KEYS or "uhid" in low_h or "mrn" in low_h or "patient" in low_h:
                    if id_col_idx is None:
                        id_col_idx = idx
                elif fallback_id_col_idx is None:
                    fallback_id_col_idx = idx
            if prof.header_has_pii:
                stats.add("PATIENT_NAME")
                out_header.append(f"col_{idx + 1}")
            else:
                out_header.append(prof.raw_name)
        if id_col_idx is None:
            id_col_idx = fallback_id_col_idx
        if header or body:
            writer.writerow(out_header)

        for i, row in enumerate(body):
            loc = f"row:{i}"
            plan = plans[f"{rel}::{loc}"]
            if clean_doc:
                cleaned_row = [clean_free_text(c, stats, clean_only_direct_ids=True, hospital_hint=item.hospital_hint) for c in row]
            else:
                cleaned_row = []
                for j, cell in enumerate(row):
                    if j < len(profiles):
                        prof = profiles[j]
                        cleaned_row.append(
                            sanitize_structured_cell(
                                cell, prof.role, prof.date_order, plan, stats, hospital_hint=item.hospital_hint
                            )
                        )
                    else:
                        cleaned_row.append(clean_free_text(cell, stats, plan=plan, hospital_hint=item.hospital_hint))
                # Ensure linked pseudonym appears inside the exported record
                if plan.pseudonym and id_col_idx is not None and id_col_idx < len(cleaned_row):
                    cleaned_row[id_col_idx] = plan.pseudonym
                elif plan.pseudonym and cleaned_row and plan.pseudonym not in cleaned_row:
                    # Fallback if table had no MRN column: embed in last free-text/notes cell without altering column count
                    cleaned_row[-1] = f"{cleaned_row[-1]} [{plan.pseudonym}]".strip()

            writer.writerow(cleaned_row)
            rec_entry: dict[str, Any] = {
                "file": f"sanitized/{out_rel}",
                "loc": loc,
                "patient": plan.pseudonym,
            }
            if not clean_doc:
                rec_entry["status"] = _record_status(plan)
            records_out.append(rec_entry)
        return buf.getvalue()

    if item.format_type in ("json", "fhir"):
        data = item.parsed
        is_doc: bool = bool(ctx.get("is_doc", True))
        if item.format_type == "fhir":
            plan = plans[f"{rel}::doc"]
            out_obj = _sanitize_fhir_bundle(data, plan, stats, item.hospital_hint)
            records_out.append(
                {
                    "file": f"sanitized/{out_rel}",
                    "loc": "doc",
                    "patient": plan.pseudonym,
                    "status": _record_status(plan),
                }
            )
            return json.dumps(out_obj, ensure_ascii=False, indent=2) + "\n"

        if is_doc:
            plan = plans[f"{rel}::doc"]
            out_obj = _sanitize_json_iterative(data, plan, stats, item.hospital_hint, clean_doc)
            if not clean_doc and plan.pseudonym and isinstance(out_obj, dict):
                # Inject pseudonym into patient sub-object or top-level ref without changing key structure wildly
                if isinstance(out_obj.get("patient"), dict) and "hospital_uhid" in out_obj["patient"]:
                    out_obj["patient"]["hospital_uhid"] = plan.pseudonym
                elif "patient_ref" in out_obj:
                    out_obj["patient_ref"] = plan.pseudonym
                else:
                    out_obj["patient_pseudonym"] = plan.pseudonym
            rec_entry = {
                "file": f"sanitized/{out_rel}",
                "loc": "doc",
                "patient": plan.pseudonym,
            }
            if not clean_doc:
                rec_entry["status"] = _record_status(plan)
            records_out.append(rec_entry)
            return json.dumps(out_obj, ensure_ascii=False, indent=2) + "\n"

        # Multi-record JSON (e.g. opd/*.json, registration/dhh_patients.json, lab/bulk_lab_export.json)
        out_root = _sanitize_json_iterative(data, None, stats, item.hospital_hint, clean_doc)
        # Re-sanitize each record item with its specific RecordUnit PrivacyPlan
        for k, v in list(data.items()):
            if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                pointer_key = k.replace("~", "~0").replace("/", "~1")
                sanitized_list: list[Any] = []
                for i, item_dict in enumerate(v):
                    loc = f"/{pointer_key}/{i}"
                    plan = plans[f"{rel}::{loc}"]
                    cleaned_item = _sanitize_json_iterative(
                        item_dict, plan, stats, item.hospital_hint, clean_doc
                    )
                    if not clean_doc and plan.pseudonym and isinstance(cleaned_item, dict):
                        if "patient_ref" in cleaned_item:
                            cleaned_item["patient_ref"] = plan.pseudonym
                        elif "uhid" in cleaned_item:
                            cleaned_item["uhid"] = plan.pseudonym
                        else:
                            cleaned_item["patient_pseudonym"] = plan.pseudonym
                    sanitized_list.append(cleaned_item)
                    rec_entry = {
                        "file": f"sanitized/{out_rel}",
                        "loc": loc,
                        "patient": plan.pseudonym,
                    }
                    if not clean_doc:
                        rec_entry["status"] = _record_status(plan)
                    records_out.append(rec_entry)
                out_root[k] = sanitized_list
        return json.dumps(out_root, ensure_ascii=False, indent=2) + "\n"

    # Plain text (.txt)
    plan = plans[f"{rel}::doc"]
    cleaned_txt = clean_free_text(
        item.text,
        stats,
        plan=None if clean_doc else plan,
        clean_only_direct_ids=clean_doc,
        hospital_hint=item.hospital_hint,
    )
    if not clean_doc and plan.pseudonym:
        cleaned_txt = f"Patient: {plan.pseudonym}\n{cleaned_txt}"
    rec_entry = {
        "file": f"sanitized/{out_rel}",
        "loc": "doc",
        "patient": plan.pseudonym,
    }
    if not clean_doc:
        rec_entry["status"] = _record_status(plan)
    records_out.append(rec_entry)
    return cleaned_txt
