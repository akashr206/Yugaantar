"""Privacy planning pipeline: clue extraction, identity resolution, and adaptive privacy.

Orchestrates cross-script normalization, multi-format clue extraction,
precision-first clustering, keyed HMAC pseudonyms, per-patient date shifts,
and Option C adaptive k-anonymity planning.
"""
from __future__ import annotations

from rakshak.clue_extractor import extract_clues_for_unit, infer_file_date_orders, infer_file_field_roles
from rakshak.contracts import IdentityClues, PolicyConfig, PrivacyPlan, RecordUnit
from rakshak.linkage import link_records
from rakshak.privacy import build_privacy_plans


def plan_privacy_full(
    units: list[RecordUnit],
    policy: PolicyConfig,
    secret: bytes,
) -> tuple[dict[str, PrivacyPlan], dict[tuple[str, str], str], list[IdentityClues]]:
    """Run clue extraction, identity linkage, and adaptive k-anonymity planning."""
    date_orders = infer_file_date_orders(units)
    field_roles = infer_file_field_roles(units)
    clues = [extract_clues_for_unit(u, date_orders, field_roles=field_roles) for u in units]
    cluster_map = link_records(clues)
    plans = build_privacy_plans(clues, cluster_map, policy, secret)
    return plans, date_orders, clues


def plan_privacy_with_date_orders(
    units: list[RecordUnit],
    policy: PolicyConfig,
    secret: bytes,
) -> tuple[dict[str, PrivacyPlan], dict[tuple[str, str], str]]:
    """Run clue extraction, identity linkage, and adaptive k-anonymity planning."""
    plans, date_orders, _ = plan_privacy_full(units, policy, secret)
    return plans, date_orders


def plan_privacy_release(
    units: list[RecordUnit],
    policy: PolicyConfig,
    secret: bytes,
) -> dict[str, PrivacyPlan]:
    """Compute per-record PrivacyPlan assignments for a batch of RecordUnits."""
    plans, _ = plan_privacy_with_date_orders(units, policy, secret)
    return plans
