"""Shared data contracts for the Rakshak privacy firewall pipeline.

Defines typed record handoff boundaries across ingestion, identity linkage,
privacy planning, and format-preserving sanitization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PolicyConfig:
    """Typed configuration loaded from config/policy.yaml."""

    age_band_years: int = 5
    date_shift_max_days: int = 180
    min_group_size: int = 5
    rare_condition_threshold: int = 3
    pin_prefix_digits: int = 3
    max_age_cap: int = 85
    adaptive_widening: bool = True

    @classmethod
    def from_yaml(cls, path: Path) -> "PolicyConfig":
        """Load simple key: value pairs without requiring PyYAML at import time."""
        raw: dict[str, Any] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                cleaned = line.split("#", 1)[0].strip()
                if ":" not in cleaned:
                    continue
                k, v = (part.strip() for part in cleaned.split(":", 1))
                low = v.lower()
                if low in ("true", "yes", "on"):
                    raw[k] = True
                elif low in ("false", "no", "off"):
                    raw[k] = False
                elif v.lstrip("-").isdigit():
                    raw[k] = int(v)
                else:
                    raw[k] = v
        return cls(
            age_band_years=int(raw.get("age_band_years", 5)),
            date_shift_max_days=int(raw.get("date_shift_max_days", 180)),
            min_group_size=int(raw.get("min_group_size", 5)),
            rare_condition_threshold=int(raw.get("rare_condition_threshold", 3)),
            pin_prefix_digits=int(raw.get("pin_prefix_digits", 3)),
            max_age_cap=int(raw.get("max_age_cap", 85)),
            adaptive_widening=bool(raw.get("adaptive_widening", True)),
        )


@dataclass(frozen=True)
class DateMention:
    """A parsed or raw clinical/demographic date extracted from a record."""

    raw_text: str
    iso_date: str | None = None      # YYYY-MM-DD when resolved
    role: str = "clinical"           # dob | admission | discharge | visit | registration | clinical
    date_order: str = "dmy"          # dmy | mdy | iso
    field_path: str = ""             # e.g. "col:Reg Date", "/visit_date", "line:9"


@dataclass(frozen=True)
class RecordUnit:
    """Normalized record representation produced by ingestion/profiling.

    unit_id uniquely identifies one scorable record as "<file_rel>::<loc>".
    """

    unit_id: str
    file_rel: str
    loc: str                         # "row:N", "/json/pointer", or "doc"
    format_type: str                 # "csv" | "tsv" | "json" | "fhir" | "txt"
    hospital_hint: str = ""          # normalized facility slug inferred from document or path
    raw_text: str = ""
    structured_fields: dict[str, Any] = field(default_factory=dict)
    extracted_dates: tuple[DateMention, ...] = ()
    metadata_hints: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class NameClue:
    """Normalized name representation used during identity resolution."""

    raw: str
    latin_tokens: tuple[str, ...]
    phonetic_tokens: tuple[str, ...]
    role: str = "patient"            # patient | relative


@dataclass(frozen=True)
class IdentityClues:
    """Extracted identity clue vector used for record linkage and privacy planning."""

    unit_id: str
    file_rel: str
    loc: str
    patient_names: tuple[NameClue, ...] = ()
    relative_names: tuple[NameClue, ...] = ()
    phones: tuple[str, ...] = ()
    aadhaars: tuple[str, ...] = ()
    aadhaar_last4: tuple[str, ...] = ()
    abhas: tuple[str, ...] = ()
    abha_addresses: tuple[str, ...] = ()
    mrns: tuple[str, ...] = ()       # normalized hospital-scoped or full MRN/UHID/IPD
    emails: tuple[str, ...] = ()
    pans: tuple[str, ...] = ()
    insurance_ids: tuple[str, ...] = ()
    dob: str | None = None           # YYYY-MM-DD
    age: int | None = None
    gender: str | None = None        # "M" | "F" | "O"
    village_norm: str | None = None
    district_norm: str | None = None
    pin: str | None = None
    hospital: str = ""
    dates: tuple[DateMention, ...] = ()
    conditions: tuple[str, ...] = () # ICD-10 codes or canonical condition tokens


@dataclass(frozen=True)
class PrivacyPlan:
    """Per-record privacy directive returned by the linkage and privacy planner.

    Specifies the keyed pseudonym, per-patient date shift, and quasi-identifier
    generalization tier to apply during sanitization and export.
    """

    unit_id: str
    cluster_key: str | None          # deterministic non-secret cluster id (None if unlinked)
    pseudonym: str | None            # HMAC "PSN-XXXXXX" or None when ambiguous
    date_shift_days: int             # non-zero per-patient (or per-record) offset
    age_band: str | None             # truthful band e.g. "45-49", "85+", or widened "80-99"
    keep_district: bool = True
    keep_pin_prefix: bool = True
    generalization_level: str = "standard"  # "standard" | "widened" | "suppressed"
    keep_gender: bool = True
