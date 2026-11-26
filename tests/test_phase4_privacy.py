"""Phase 4 unit tests for Option C adaptive k-anonymity, date shifting, and age bands."""
from __future__ import annotations

import datetime as dt
import unittest

from rakshak.contracts import IdentityClues, PolicyConfig
from rakshak.privacy import (
    apply_privacy_to_text,
    build_privacy_plans,
    format_age_band,
    generalize_pin,
    make_date_shift_days,
    make_pseudonym,
    shift_date_token,
)


class TestPhase4Privacy(unittest.TestCase):
    def test_hmac_pseudonyms_and_shifts_depend_on_secret(self) -> None:
        p1 = make_pseudonym("ID|snj26064127", b"secret-1")
        p2 = make_pseudonym("ID|snj26064127", b"secret-1")
        p3 = make_pseudonym("ID|snj26064127", b"secret-2")
        self.assertEqual(p1, p2)
        self.assertNotEqual(p1, p3)
        self.assertTrue(p1.startswith("PSN-"))

        s1 = make_date_shift_days("ID|snj26064127", b"secret-1", 180)
        s2 = make_date_shift_days("ID|snj26064127", b"secret-2", 180)
        self.assertNotEqual(s1, 0)
        self.assertNotEqual(s1, s2)

    def test_date_shifting_preserves_exact_stay_length(self) -> None:
        adm = "23/07/2026"
        dis = "28/07/2026"
        shift = -47
        adm_s = shift_date_token(adm, shift, "dmy")
        dis_s = shift_date_token(dis, shift, "dmy")
        d1 = dt.datetime.strptime(adm_s, "%d/%m/%Y").date()
        d2 = dt.datetime.strptime(dis_s, "%d/%m/%Y").date()
        self.assertEqual((d2 - d1).days, 5)
        self.assertNotEqual(adm_s, adm)

    def test_truthful_age_bands(self) -> None:
        for age in (0, 4, 5, 48, 84, 85, 95):
            band = format_age_band(age, width=5, cap=85)
            if band.endswith("+"):
                lo = int(band[:-1])
                self.assertGreaterEqual(age, lo)
            else:
                lo, hi = (int(x) for x in band.split("-"))
                self.assertLessEqual(lo, age)
                self.assertLessEqual(age, hi)

    def test_adaptive_k_anonymity_option_c(self) -> None:
        policy = PolicyConfig(age_band_years=5, min_group_size=3, rare_condition_threshold=1, adaptive_widening=True)
        clues = []
        cmap = {}
        # Group of 3 common patients (45-49, M, lucknow, common E11.9) -> standard
        for i in range(3):
            uid = f"f{i}::doc"
            clues.append(IdentityClues(uid, "f", "doc", age=46 + (i % 2), gender="M", district_norm="lckn", conditions=("E11.9",)))
            cmap[uid] = f"ID|common_{i}"
        # 1 rare-condition elderly outlier (95, M, bahraich, rare A82.9) -> suppressed QI
        rare_uid = "rare::doc"
        clues.append(IdentityClues(rare_uid, "rare", "doc", age=95, gender="M", district_norm="bhrch", conditions=("A82.9",)))
        cmap[rare_uid] = "ID|rare_0"

        plans = build_privacy_plans(clues, cmap, policy, b"sec")
        self.assertEqual(plans["f0::doc"].generalization_level, "standard")
        self.assertEqual(plans["f0::doc"].age_band, "45-49")
        self.assertTrue(plans["f0::doc"].keep_district)
        self.assertTrue(plans["f0::doc"].keep_pin_prefix)

        self.assertEqual(plans[rare_uid].generalization_level, "suppressed")
        self.assertEqual(plans[rare_uid].age_band, "80+")
        self.assertFalse(plans[rare_uid].keep_district)
        self.assertFalse(plans[rare_uid].keep_pin_prefix)

    def test_text_transform_preserves_doses_and_shifts_dates(self) -> None:
        policy = PolicyConfig(min_group_size=1)
        c = IdentityClues("d::doc", "d", "doc", age=39, gender="M", district_norm="stpr")
        plan = build_privacy_plans([c], {"d::doc": "ID|x"}, policy, b"sec")["d::doc"]
        txt = (
            "Patient Name: X   Age/Sex: 39 Yrs/M\n"
            "Date of Admission: 13/01/2026\n"
            "Date of Discharge: 18/01/2026\n"
            "Tab Paracetamol 650 mg 1-1-1 x 5 days\n"
            "Random Blood Sugar 125 mg/dL (70-140)\n"
        )
        new_txt, changed = apply_privacy_to_text(txt, plan, "dmy")
        self.assertTrue(changed)
        self.assertIn("Age/Sex: 35-39 Yrs/M", new_txt)
        # Running a second pass must not double-band "35-39"
        new_txt2, _ = apply_privacy_to_text(new_txt, plan, "dmy")
        self.assertNotIn("35-39-39", new_txt2)
        self.assertIn("650 mg 1-1-1 x 5 days", new_txt)
        self.assertIn("(70-140)", new_txt)
        self.assertNotIn("13/01/2026", new_txt)
        self.assertEqual(generalize_pin("226183", True, 3), "226***")
        self.assertEqual(generalize_pin("226183", False, 3), "[PIN]")


if __name__ == "__main__":
    unittest.main()
