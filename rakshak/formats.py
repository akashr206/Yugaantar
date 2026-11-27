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
    header_contains_pii,
    is_clean_reference_file,
    parse_date_string,
    profile_records_keys,
    profile_table,
    try_decode_base64_text,
)

_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_RECORD_ID_KEYS = (
    "uhid", "patient_ref", "hospital_uhid", "patientid", "patient_id",
    "reg_no", "mrn", "ipd_no", "opd_no", "member_id", "subscriber_id",
)
_RELATIVE_CONTAINER_KEYS = frozenset({
    "guardian", "kin", "nominee", "next_of_kin", "spouse", "attendant", "relative", "father", "mother",
})
_FHIR_SUBJECT_RESOURCE_TYPES = frozenset({
    "Composition", "Observation", "Condition", "Encounter", "DiagnosticReport",
    "MedicationRequest", "Procedure", "DocumentReference", "AllergyIntolerance",
    "Specimen", "CarePlan", "Immunization", "ImagingStudy", "ServiceRequest",
    "RelatedPerson",
})
_SAFE_PATH_TOKENS = frozenset({
    "data", "srv", "var", "tmp", "opt", "home", "usr", "export", "exports", "backup", "backups",
    "archive", "archives", "raw", "clean", "records", "patient", "patients", "report", "reports",
    "billing", "claims", "registration", "registry", "lab", "labs", "discharge", "prescriptions",
    "prescription", "fhir", "bundle", "clinical", "ward", "census", "drug", "formulary",
    "reference", "ranges", "protocol", "guideline", "summary", "health", "hospital", "clinic",
    "notes", "note", "helpdesk", "inbox", "outbox", "files", "file", "json", "csv", "tsv",
})


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
    # Infer unambiguous DMY vs MDY across string values in this JSON object
    dmy_v = mdy_v = 0
    raw_pairs: list[tuple[str, str, str]] = []
    stack: list[tuple[str, Any]] = [("", obj)]
    while stack:
        prefix, cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                path = f"{prefix}/{k}"
                if isinstance(v, str) and v.strip():
                    low = k.lower()
                    if any(t in low for t in ("dob", "birthdate", "date", "collected", "timestamp", "recordeddate", "authoredon", "start", "end")):
                        raw_pairs.append((path, low, v))
                        parts = re.split(r"[-/.]", v.strip().translate(_DEVA_DIGITS))
                        if len(parts) == 3 and len(parts[0]) != 4 and all(p.isdigit() for p in parts[:2]):
                            p1, p2 = int(parts[0]), int(parts[1])
                            if p1 > 12:
                                dmy_v += 1
                            elif p2 > 12:
                                mdy_v += 1
                elif isinstance(v, (dict, list)):
                    stack.append((path, v))
        elif isinstance(cur, list):
            for idx, item in enumerate(cur):
                if isinstance(item, (dict, list)):
                    stack.append((f"{prefix}/{idx}", item))
    obj_order = "mdy" if mdy_v > dmy_v else "dmy"
    for path, low, v in raw_pairs:
        order = "iso" if "-" in v[:5] else obj_order
        parsed = parse_date_string(v, preferred_order=order)
        if parsed:
            role = "dob" if ("dob" in low or "birth" in low) else (
                "admission" if low == "start" else ("discharge" if low == "end" else "clinical")
            )
            mentions.append(
                DateMention(
                    raw_text=v,
                    iso_date=parsed.isoformat(),
                    role=role,
                    date_order=order,
                    field_path=path,
                )
            )
    return tuple(mentions)


def _is_doc_json_container(data: Any) -> bool:
    if not isinstance(data, dict):
        return True
    if data.get("resourceType") == "Bundle":
        return True
    # Single-patient diagnostic reports (top-level "patient" dict) or single-ticket docs
    if isinstance(data.get("patient"), dict):
        return True
    list_keys = [k for k, v in data.items() if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)]
    if "panels" in list_keys and len(list_keys) == 1:
        return True
    if isinstance(data.get("tickets"), list) and len(data["tickets"]) <= 1 and len(list_keys) <= 1:
        return True
    return not list_keys


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
    file_date_orders: dict[str, str] | None = None,
    cluster_redact: tuple[tuple[str, str], ...] = (),
    path_redact_tokens: frozenset[str] = frozenset(),
    field_roles: dict[str, Any] | None = None,
    key_path: str = "",
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
    if prof.role == ROLE_UNKNOWN and field_roles:
        pooled_prof = (field_roles.get(key_path) if key_path else None) or field_roles.get(key)
        if pooled_prof is not None and pooled_prof.role != ROLE_UNKNOWN:
            prof = pooled_prof
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

    # Semantic overrides for nested JSON person/guardian/kin sub-fields
    if low_parent == "name" and low_key in ("local", "full"):
        return sanitize_structured_cell(val, ROLE_PATIENT_NAME, "dmy", plan, stats, hospital_hint=hospital_hint)
    if low_parent in _RELATIVE_CONTAINER_KEYS and low_key == "name":
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
        cleaned_inner = clean_free_text(
            decoded_b64, stats, plan=plan, hospital_hint=hospital_hint, cluster_redact=cluster_redact
        )
        return encode_base64_text(cleaned_inner)

    # _export_meta.source_path / file_path generic segment scrub (works for any folder depth or extension)
    if low_key in ("source_path", "export_path", "file_path"):
        parts = re.split(r"([\\/]+)", val)
        out_parts: list[str] = []
        for idx, seg in enumerate(parts):
            if not seg or re.fullmatch(r"[\\/]+", seg) or re.fullmatch(r"[A-Za-z]:", seg):
                out_parts.append(seg)
                continue
            is_last = idx == len(parts) - 1
            stem, ext = (seg.rsplit(".", 1)[0], "." + seg.rsplit(".", 1)[1]) if (is_last and "." in seg) else (seg, "")
            seg_toks = [w.lower() for w in re.split(r"[^A-Za-z0-9]+", stem) if len(w) >= 3 and w.isalpha()]
            has_known_tok = bool(path_redact_tokens) and any(t in path_redact_tokens for t in seg_toks)
            cleaned_seg = clean_free_text(stem, stats, plan=None, hospital_hint=hospital_hint, allow_ml_fallback=False)
            if cleaned_seg != stem:
                out_parts.append("[REDACTED]" + ext)
            elif (
                has_known_tok
                or header_contains_pii(stem)
                or re.search(r"(?:[A-Z][a-z]+_[A-Z][a-z]+|\d{6,})", stem)
                or (is_last and ("_" in stem or "-" in stem) and len(seg_toks) >= 2 and any(t not in _SAFE_PATH_TOKENS for t in seg_toks))
            ):
                stats.add("PATIENT_NAME")
                out_parts.append("[REDACTED]" + ext)
            else:
                out_parts.append(seg)
        return "".join(out_parts)

    # JSON keys that hold patient identifiers directly
    if prof.role == ROLE_MRN and (
        low_key in _RECORD_ID_KEYS
        or "mrn" in low_key
        or "uhid" in low_key
        or (field_roles is not None and not any(x in low_key for x in ("bill", "sample")))
    ):
        stats.add("MRN")
        return plan.pseudonym if (plan and plan.pseudonym) else "[MRN]"

    if prof.role == ROLE_PROTECTED_CLINICAL:
        return clean_free_text(val, stats, plan=None, clean_only_direct_ids=True, hospital_hint=hospital_hint)

    if prof.role != ROLE_UNKNOWN:
        resolved_order = prof.date_order
        if file_date_orders and prof.date_order != "iso":
            low_k = key.lower()
            if not any(h in low_k for h in ("mm/dd", "dd/mm", "yyyy-mm-dd", "mdy", "dmy", "iso")):
                resolved_order = file_date_orders.get(key) or file_date_orders.get("__doc__") or prof.date_order
        return sanitize_structured_cell(
            val, prof.role, resolved_order, plan, stats, hospital_hint=hospital_hint, cluster_redact=cluster_redact
        )

    doc_order = file_date_orders.get("__doc__") if file_date_orders else None
    return clean_free_text(
        val,
        stats,
        plan=plan,
        hospital_hint=hospital_hint,
        preferred_order=doc_order,
        cluster_redact=cluster_redact,
    )


@lru_cache(maxsize=4096)
def _clean_json_key_cached(k: str, hospital_hint: str) -> tuple[str, tuple[tuple[str, int], ...]]:
    tmp_stats = DetectionStats()
    c = clean_free_text(k, tmp_stats, plan=None, hospital_hint=hospital_hint, allow_ml_fallback=False)
    if c == k and header_contains_pii(k):
        tmp_stats.add("PATIENT_NAME")
        c = "[PATIENT_NAME]"
    return c, tuple(tmp_stats.by_type.items())


def _sanitize_json_iterative(
    root: Any,
    plan: PrivacyPlan | None,
    stats: DetectionStats,
    hospital_hint: str,
    clean_doc: bool,
    file_date_orders: dict[str, str] | None = None,
    cluster_redact: tuple[tuple[str, str], ...] = (),
    path_redact_tokens: frozenset[str] = frozenset(),
    field_roles: dict[str, Any] | None = None,
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
        return _sanitize_json_scalar(
            "", root, plan, stats, hospital_hint, clean_doc,
            file_date_orders=file_date_orders,
            cluster_redact=cluster_redact,
            path_redact_tokens=path_redact_tokens,
            field_roles=field_roles,
        )

    new_root: Any = {} if isinstance(root, dict) else []
    # Stack items: (src_node, dst_node, parent_key, key_path, inside_protected_resource)
    stack: list[tuple[Any, Any, str, str, bool]] = [(root, new_root, "", "", False)]

    while stack:
        src, dst, pkey, cpath, in_prac = stack.pop()
        if isinstance(src, dict):
            rtype = src.get("resourceType")
            is_prac_obj = in_prac or rtype in ("Practitioner", "Organization")
            for raw_k, v in src.items():
                rel_path = f"{cpath}.{raw_k}" if cpath else raw_k
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
                    stack.append((v, child, raw_k, rel_path, is_prac_obj))
                elif isinstance(v, list):
                    child_list: list[Any] = []
                    dst[ck] = child_list
                    stack.append((v, child_list, raw_k, rel_path, is_prac_obj))
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
                        file_date_orders=file_date_orders,
                        cluster_redact=cluster_redact,
                        path_redact_tokens=path_redact_tokens,
                        field_roles=field_roles,
                        key_path=rel_path,
                    )
        elif isinstance(src, list):
            arr_path = f"{cpath}[]" if cpath else "[]"
            for item in src:
                if isinstance(item, dict):
                    child_d: dict[str, Any] = {}
                    dst.append(child_d)
                    stack.append((item, child_d, pkey, arr_path, in_prac))
                elif isinstance(item, list):
                    child_l: list[Any] = []
                    dst.append(child_l)
                    stack.append((item, child_l, pkey, arr_path, in_prac))
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
                            file_date_orders=file_date_orders,
                            cluster_redact=cluster_redact,
                            path_redact_tokens=path_redact_tokens,
                            field_roles=field_roles,
                            key_path=arr_path,
                        )
                    )
    return new_root


def _sanitize_fhir_bundle(
    bundle: dict[str, Any],
    plan: PrivacyPlan,
    stats: DetectionStats,
    hospital_hint: str,
    file_date_orders: dict[str, str] | None = None,
    cluster_redact: tuple[tuple[str, str], ...] = (),
    path_redact_tokens: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Sanitize a FHIR R4 Document Bundle while keeping references valid and Patient PSN injected."""
    cleaned = _sanitize_json_iterative(
        bundle,
        plan,
        stats,
        hospital_hint,
        clean_doc=False,
        file_date_orders=file_date_orders,
        cluster_redact=cluster_redact,
        path_redact_tokens=path_redact_tokens,
    )
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
        if rtype in _FHIR_SUBJECT_RESOURCE_TYPES:
            for ref_key in ("subject", "patient"):
                subj = res.get(ref_key)
                if isinstance(subj, dict) and isinstance(subj.get("display"), str):
                    stats.add("PATIENT_NAME")
                    subj["display"] = psn if psn else "[PATIENT_NAME]"
        if rtype == "RelatedPerson":
            stats.add("RELATIVE_NAME")
            res["name"] = [{"text": "[RELATIVE_NAME]"}]
            res.pop("birthDate", None)
            if isinstance(res.get("telecom"), list):
                for tc in res["telecom"]:
                    if isinstance(tc, dict) and isinstance(tc.get("value"), str):
                        sys_low = str(tc.get("system", "")).lower()
                        kind = "EMAIL" if sys_low == "email" or "@" in tc["value"] else "PHONE"
                        stats.add(kind)
                        tc["value"] = f"[{kind}]"
            if isinstance(res.get("address"), list):
                for a in res["address"]:
                    if isinstance(a, dict):
                        a.pop("line", None)
                        a.pop("city", None)
                        a.pop("text", None)
                        if not plan.keep_district:
                            a["district"] = "[DISTRICT]"
                        if "postalCode" in a:
                            pc = str(a["postalCode"]).translate(_DEVA_DIGITS)
                            a["postalCode"] = f"{pc[:3]}***" if (plan.keep_pin_prefix and len(pc) >= 3) else "[PIN]"
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
            # Truncate Patient.birthDate to birth year (YYYY) on standard tier (never shift DOB); omit on widened/suppressed tiers
            if "birthDate" in res:
                if plan.generalization_level != "standard":
                    res.pop("birthDate", None)
                else:
                    m_yr = re.match(r"^(\d{4})\b", str(res["birthDate"]).strip())
                    if m_yr:
                        res["birthDate"] = m_yr.group(1)
                        stats.generalized += 1
                    else:
                        res.pop("birthDate", None)
            if not getattr(plan, "keep_gender", True):
                res["gender"] = "unknown"
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
                        a.pop("district", None)
    return cleaned


def build_cluster_redact_context(
    clues: list[Any],
    plans: dict[str, PrivacyPlan],
) -> tuple[dict[str, tuple[tuple[str, str], ...]], dict[str, frozenset[str]]]:
    """Precompute per-unit cluster redaction surfaces (names/relatives/suppressed QIs) & path tokens."""
    cluster_names: dict[str, set[tuple[str, str]]] = {}
    file_tokens: dict[str, set[str]] = {}
    for c in clues:
        plan = plans.get(c.unit_id)
        ckey = (plan.cluster_key if plan and plan.cluster_key else None) or c.unit_id
        bucket = cluster_names.setdefault(ckey, set())
        ftoks = file_tokens.setdefault(c.file_rel, set())
        for nc in c.patient_names:
            raw = nc.raw.strip()
            if len(raw) >= 4 and (" " in raw or len(nc.latin_tokens) >= 2 or len(raw) >= 5):
                bucket.add(("PATIENT_NAME", raw))
            for tok in nc.latin_tokens:
                if len(tok) >= 4:
                    ftoks.add(tok)
        for rc in c.relative_names:
            raw = rc.raw.strip()
            if len(raw) >= 4:
                bucket.add(("RELATIVE_NAME", raw))
            for tok in rc.latin_tokens:
                if len(tok) >= 4:
                    ftoks.add(tok)
        if c.district_norm and len(c.district_norm) >= 4:
            ftoks.add(c.district_norm)

    unit_redact: dict[str, tuple[tuple[str, str], ...]] = {}
    for c in clues:
        plan = plans.get(c.unit_id)
        ckey = (plan.cluster_key if plan and plan.cluster_key else None) or c.unit_id
        pairs = set(cluster_names.get(ckey, set()))
        if plan is not None and not plan.keep_district and c.district_norm and len(c.district_norm) >= 4:
            pairs.add(("DISTRICT", c.district_norm))
        if plan is not None and not plan.keep_pin_prefix and c.pin and len(c.pin) == 6:
            pairs.add(("PIN", c.pin))
        unit_redact[c.unit_id] = tuple(sorted(pairs))

    return unit_redact, {k: frozenset(v) for k, v in file_tokens.items()}


def render_sanitized_file(
    item: IngestResult,
    ctx: dict[str, Any],
    plans: dict[str, PrivacyPlan],
    records_out: list[dict[str, Any]],
    stats: DetectionStats,
    date_orders: dict[tuple[str, str], str] | None = None,
    clues: list[Any] | None = None,
    cluster_redact_context: tuple[dict[str, tuple[tuple[str, str], ...]], dict[str, frozenset[str]]] | None = None,
) -> str:
    """Render one sanitized file and append its records.jsonl entries."""
    rel = item.file_rel
    out_rel = item.clean_rel
    clean_doc: bool = bool(ctx.get("clean_doc", False))
    file_orders = {fk: ord_v for (frel, fk), ord_v in (date_orders or {}).items() if frel == rel}
    if cluster_redact_context is None and clues is not None:
        cluster_redact_context = build_cluster_redact_context(clues, plans)
    unit_redact_map, file_path_tok_map = cluster_redact_context or ({}, {})
    path_toks = file_path_tok_map.get(rel, frozenset())

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
                        col_order = file_orders.get(prof.raw_name, prof.date_order)
                        cleaned_row.append(
                            sanitize_structured_cell(
                                cell,
                                prof.role,
                                col_order,
                                plan,
                                stats,
                                hospital_hint=item.hospital_hint,
                                cluster_redact=unit_redact_map.get(f"{rel}::{loc}", ()),
                            )
                        )
                    else:
                        cleaned_row.append(
                            clean_free_text(
                                cell,
                                stats,
                                plan=plan,
                                hospital_hint=item.hospital_hint,
                                cluster_redact=unit_redact_map.get(f"{rel}::{loc}", ()),
                            )
                        )
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
            out_obj = _sanitize_fhir_bundle(
                data,
                plan,
                stats,
                item.hospital_hint,
                file_date_orders=file_orders,
                cluster_redact=unit_redact_map.get(f"{rel}::doc", ()),
                path_redact_tokens=path_toks,
            )
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
            out_obj = _sanitize_json_iterative(
                data,
                plan,
                stats,
                item.hospital_hint,
                clean_doc,
                file_date_orders=file_orders,
                cluster_redact=unit_redact_map.get(f"{rel}::doc", ()),
                path_redact_tokens=path_toks,
            )
            if not clean_doc and plan.pseudonym and isinstance(out_obj, dict):
                # Inject pseudonym into existing patient/MRN key first, else add patient_pseudonym
                if isinstance(out_obj.get("patient"), dict):
                    pt_d = out_obj["patient"]
                    id_k = next((rk for rk in _RECORD_ID_KEYS if rk in pt_d), None)
                    if id_k:
                        pt_d[id_k] = plan.pseudonym
                    else:
                        pt_d["patient_pseudonym"] = plan.pseudonym
                else:
                    top_k = next((rk for rk in _RECORD_ID_KEYS if rk in out_obj), None)
                    if top_k:
                        out_obj[top_k] = plan.pseudonym
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
        out_root = _sanitize_json_iterative(
            data,
            None,
            stats,
            item.hospital_hint,
            clean_doc,
            file_date_orders=file_orders,
            path_redact_tokens=path_toks,
        )
        # Re-sanitize each record item with its specific RecordUnit PrivacyPlan & pooled array roles
        for k, v in list(data.items()):
            if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                arr_roles = profile_records_keys(v, hospital_hint=item.hospital_hint)
                pointer_key = k.replace("~", "~0").replace("/", "~1")
                sanitized_list: list[Any] = []
                for i, item_dict in enumerate(v):
                    loc = f"/{pointer_key}/{i}"
                    plan = plans[f"{rel}::{loc}"]
                    cleaned_item = _sanitize_json_iterative(
                        item_dict,
                        plan,
                        stats,
                        item.hospital_hint,
                        clean_doc,
                        file_date_orders=file_orders,
                        cluster_redact=unit_redact_map.get(f"{rel}::{loc}", ()),
                        path_redact_tokens=path_toks,
                        field_roles=arr_roles,
                    )
                    if not clean_doc and plan.pseudonym and isinstance(cleaned_item, dict):
                        id_k = next((rk for rk in _RECORD_ID_KEYS if rk in cleaned_item), None)
                        if id_k is None:
                            id_k = next(
                                (
                                    cand_k
                                    for cand_k in cleaned_item
                                    if arr_roles.get(cand_k) and arr_roles[cand_k].role == ROLE_MRN
                                    and not any(x in cand_k.lower() for x in ("bill", "sample"))
                                ),
                                None,
                            )
                        if id_k:
                            cleaned_item[id_k] = plan.pseudonym
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
        preferred_order=file_orders.get("__doc__"),
        cluster_redact=() if clean_doc else unit_redact_map.get(f"{rel}::doc", ()),
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
