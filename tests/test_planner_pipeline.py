"""Unit tests for the integrated privacy planner facade (`rakshak.planner`)."""
from __future__ import annotations

import unittest

from rakshak.contracts import PolicyConfig, RecordUnit
from rakshak.planner import plan_privacy_release, plan_privacy_with_date_orders


class TestPlannerPipeline(unittest.TestCase):
    def _sample_units(self) -> list[RecordUnit]:
        return [
            RecordUnit(
                unit_id="reg/a.csv::row:0",
                file_rel="reg/a.csv",
                loc="row:0",
                format_type="csv",
                hospital_hint="snj",
                structured_fields={
                    "UHID": "SNJ/26/064127",
                    "Patient Name": "Mahesh Chauhan",
                    "Age": "48",
                    "Gender": "M",
                    "District": "Rae Bareli",
                    "PIN": "229073",
                    "Reg Date": "30/06/2025",
                },
            ),
            RecordUnit(
                unit_id="notes/n1.txt::doc",
                file_rel="notes/n1.txt",
                loc="doc",
                format_type="txt",
                hospital_hint="snj",
                raw_text="Patient Name: महेश चौहान\nUHID: SNJ/26/064127\nDate: 15/08/2026\n",
            ),
        ]

    def test_plan_privacy_with_date_orders(self) -> None:
        plans, date_orders = plan_privacy_with_date_orders(
            self._sample_units(), PolicyConfig(), b"planner-test-secret"
        )
        self.assertEqual(date_orders[("reg/a.csv", "Reg Date")], "dmy")
        p1 = plans["reg/a.csv::row:0"]
        p2 = plans["notes/n1.txt::doc"]
        self.assertIsNotNone(p1.pseudonym)
        self.assertEqual(p1.pseudonym, p2.pseudonym)
        self.assertEqual(p1.date_shift_days, p2.date_shift_days)

    def test_plan_privacy_release_matches_facade(self) -> None:
        units = self._sample_units()
        secret = b"planner-test-secret"
        plans_only = plan_privacy_release(units, PolicyConfig(), secret)
        plans_with_orders, _ = plan_privacy_with_date_orders(units, PolicyConfig(), secret)
        self.assertEqual(set(plans_only.keys()), set(plans_with_orders.keys()))
        for uid, plan in plans_only.items():
            self.assertEqual(plan.pseudonym, plans_with_orders[uid].pseudonym)
            self.assertEqual(plan.date_shift_days, plans_with_orders[uid].date_shift_days)
            self.assertEqual(plan.age_band, plans_with_orders[uid].age_band)


if __name__ == "__main__":
    unittest.main()
