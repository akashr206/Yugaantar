"""Phase 0 contract and harness smoke tests."""
from __future__ import annotations

import unittest
from pathlib import Path

from rakshak.contracts import DateMention, IdentityClues, PolicyConfig, PrivacyPlan, RecordUnit


class TestPhase0Contracts(unittest.TestCase):
    def test_policy_loads_option_c_defaults(self) -> None:
        policy = PolicyConfig.from_yaml(Path(__file__).resolve().parents[1] / "config" / "policy.yaml")
        self.assertEqual(policy.age_band_years, 5)
        self.assertEqual(policy.date_shift_max_days, 180)
        self.assertEqual(policy.min_group_size, 5)
        self.assertEqual(policy.rare_condition_threshold, 3)
        self.assertEqual(policy.pin_prefix_digits, 3)
        self.assertEqual(policy.max_age_cap, 85)
        self.assertTrue(policy.adaptive_widening)

    def test_contract_dataclasses_frozen(self) -> None:
        dm = DateMention(raw_text="12/04/1977", iso_date="1977-04-12", role="dob", date_order="dmy")
        ru = RecordUnit(
            unit_id="registration/sanjeevani_registration.csv::row:0",
            file_rel="registration/sanjeevani_registration.csv",
            loc="row:0",
            format_type="csv",
            hospital_hint="snj",
            extracted_dates=(dm,),
        )
        ic = IdentityClues(unit_id=ru.unit_id, file_rel=ru.file_rel, loc=ru.loc, age=48, gender="M")
        pp = PrivacyPlan(unit_id=ru.unit_id, cluster_key=None, pseudonym=None, date_shift_days=14, age_band="45-49")
        self.assertEqual(ic.age, 48)
        self.assertEqual(pp.age_band, "45-49")


if __name__ == "__main__":
    unittest.main()
