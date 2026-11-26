"""Single-call Person 2 facade for Sync 2 integration into main.py. [Person 2]"""
from __future__ import annotations

from rakshak.clue_extractor import extract_clues_for_unit, infer_file_date_orders
from rakshak.contracts import PolicyConfig, PrivacyPlan, RecordUnit
from rakshak.linkage import link_records
from rakshak.privacy import build_privacy_plans


def run_person2_pipeline(
    units: list[RecordUnit],
    policy: PolicyConfig,
    secret: bytes,
) -> tuple[dict[str, PrivacyPlan], dict[tuple[str, str], str]]:
    """Run clue extraction, identity resolution, and Option C privacy planning.

    Returns:
        plans: mapping {unit_id: PrivacyPlan}
        date_orders: mapping {(file_rel, field_key_or___doc__): "dmy"|"mdy"|"iso"}
    """
    date_orders = infer_file_date_orders(units)
    clues = [extract_clues_for_unit(u, date_orders) for u in units]
    cluster_map = link_records(clues)
    plans = build_privacy_plans(clues, cluster_map, policy, secret)
    return plans, date_orders
