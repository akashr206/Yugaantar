"""Phase 4 unit tests for Option C adaptive k-anonymity, date shifting, and age bands."""
from __future__ import annotations

import datetime as dt
import re
import unittest
from unittest.mock import patch

from rakshak.contracts import DateMention, IdentityClues, PolicyConfig
from rakshak.detector import DetectionStats, clean_free_text, sanitize_structured_cell
from rakshak.formats import _sanitize_fhir_bundle, _sanitize_json_scalar
from rakshak.privacy import (
    _hull_age_band,
    apply_privacy_to_text,
    build_privacy_plans,
    format_age_band,
    generalize_pin,
    make_date_shift_days,
    make_pseudonym,
    shift_date_token,
)
from rakshak.schema_profiler import ROLE_AGE, ROLE_AGE_SEX, ROLE_DISTRICT, ROLE_GENDER


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
        self.assertTrue(plans["f0::doc"].keep_gender)

        self.assertEqual(plans[rare_uid].generalization_level, "suppressed")
        # Singleton rare outlier retains truthful wide band (never '[AGE]') and suppresses gender
        self.assertNotEqual(plans[rare_uid].age_band, "[AGE]")
        self.assertEqual(plans[rare_uid].age_band, "80+")
        self.assertFalse(plans[rare_uid].keep_district)
        self.assertFalse(plans[rare_uid].keep_pin_prefix)
        self.assertFalse(plans[rare_uid].keep_gender)

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

    def test_pseudonym_length_and_deterministic_collision_resolution(self) -> None:
        pid = make_pseudonym("ID|test123", b"secret-1")
        self.assertRegex(pid, r"^PSN-[0-9A-Z]{10}$")

        policy = PolicyConfig(min_group_size=1)
        c1 = IdentityClues("a::doc", "a", "doc", age=30, gender="M")
        c2 = IdentityClues("b::doc", "b", "doc", age=40, gender="F")
        cmap_1 = {"a::doc": "ID|alpha", "b::doc": "ID|beta"}

        def fake_make_pseudonym(cluster_key: str, secret: bytes) -> str:
            # Force a deterministic collision between initial alpha and beta
            if cluster_key in ("ID|alpha", "ID|beta"):
                return "PSN-COLLIDE001"
            return "PSN-RESOLVED02"

        with patch("rakshak.privacy.make_pseudonym", side_effect=fake_make_pseudonym):
            plans_1 = build_privacy_plans([c1, c2], cmap_1, policy, b"sec")
            plans_2 = build_privacy_plans([c2, c1], cmap_1, policy, b"sec")  # reversed input order
            plans_3 = build_privacy_plans([c1, c2], cmap_1, policy, b"sec")  # repeat run

        self.assertNotEqual(plans_1["a::doc"].pseudonym, plans_1["b::doc"].pseudonym)
        self.assertEqual(plans_1["a::doc"].pseudonym, plans_2["a::doc"].pseudonym)
        self.assertEqual(plans_1["b::doc"].pseudonym, plans_2["b::doc"].pseudonym)
        self.assertEqual(plans_1["b::doc"].pseudonym, plans_3["b::doc"].pseudonym)

    def test_c1_hull_age_band_and_pin_dob_subgroup_widening(self) -> None:
        # Cross-visit ages spanning band boundaries stay truthful with width <= 30
        hull = _hull_age_band([39, 41], 5, 85)
        lo, hi = (int(x) for x in hull.split("-"))
        self.assertLessEqual(lo, 39)
        self.assertGreaterEqual(hi, 41)
        self.assertLessEqual(hi - lo, 30)
        self.assertEqual(_hull_age_band([78, 81], 20, 80), "70+")

        # Within a k=5 standard cell, a unique PIN prefix / birth year is widened instead of keeping PIN/DOB
        policy = PolicyConfig(age_band_years=5, min_group_size=5, rare_condition_threshold=1, adaptive_widening=True)
        clues = []
        cmap = {}
        for i in range(5):
            uid = f"u{i}::doc"
            clues.append(
                IdentityClues(
                    uid,
                    "u",
                    "doc",
                    age=47,
                    gender="M",
                    district_norm="lckn",
                    pin="227999" if i == 0 else "226001",
                    dob=f"{1978 + (i % 2)}-05-10" if i > 0 else "1979-11-20",
                    conditions=("E11.9",),
                )
            )
            cmap[uid] = f"ID|p{i}"
        plans = build_privacy_plans(clues, cmap, policy, b"sec")
        self.assertFalse(plans["u0::doc"].keep_pin_prefix)
        self.assertEqual(plans["u0::doc"].generalization_level, "widened")
        self.assertTrue(plans["u0::doc"].keep_gender)

    def test_c1_age_and_gender_suppression_across_formats(self) -> None:
        policy = PolicyConfig(age_band_years=5, min_group_size=5, rare_condition_threshold=2, adaptive_widening=True)
        rare_f = IdentityClues(
            "rare_f::doc",
            "r",
            "doc",
            age=68,
            gender="F",
            district_norm="lckn",
            conditions=("A82.9",),
            dates=(
                DateMention("01/03/2026", "2026-03-01", role="admission"),
                DateMention("06/03/2026", "2026-03-06", role="discharge"),
            ),
        )
        # Second opposite-sex A82.9 peer inside pool triggers surgical keep_gender=False + truthful 50+ band
        rare_m = IdentityClues(
            "rare_m::doc",
            "r2",
            "doc",
            age=62,
            gender="M",
            district_norm="knpr",
            conditions=("A82.9",),
            dates=(
                DateMention("10/04/2026", "2026-04-10", role="admission"),
                DateMention("15/04/2026", "2026-04-15", role="discharge"),
            ),
        )
        plans = build_privacy_plans(
            [rare_f, rare_m],
            {"rare_f::doc": "ID|rare_f", "rare_m::doc": "ID|rare_m"},
            policy,
            b"sec",
        )
        plan = plans["rare_f::doc"]
        expected_band = plan.age_band
        self.assertIsNotNone(expected_band)
        self.assertNotEqual(expected_band, "[AGE]")
        self.assertFalse(plan.keep_gender)
        self.assertEqual(plan.generalization_level, "suppressed")

        stats = DetectionStats()
        # CSV/TSV structured cell paths
        self.assertEqual(sanitize_structured_cell("68", ROLE_AGE, "dmy", plan, stats), expected_band)
        self.assertEqual(sanitize_structured_cell("F", ROLE_GENDER, "dmy", plan, stats), "[GENDER]")
        self.assertEqual(sanitize_structured_cell("68/F", ROLE_AGE_SEX, "dmy", plan, stats), f"{expected_band}/[GENDER]")
        self.assertEqual(sanitize_structured_cell("Lucknow", ROLE_DISTRICT, "dmy", plan, stats), "[DISTRICT]")

        # Free-text paths (English, WhatsApp, Hindi) preserve clinical text & truthful wide age band
        txt = (
            "Age/Sex: 68 Yrs/F\n"
            "आयु: 68 वर्ष / महिला\n"
            "Diagnosis: Rabies, unspecified (A82.9)\n"
            "Inj Human Rabies Immunoglobulin 20 IU/kg"
        )
        cleaned = clean_free_text(txt, stats, plan=plan)
        self.assertIn(f"Age/Sex: {expected_band} Yrs/[GENDER]", cleaned)
        self.assertIn(f"आयु: {expected_band} वर्ष / [GENDER]", cleaned)
        self.assertIn("Rabies, unspecified (A82.9)", cleaned)
        self.assertIn("20 IU/kg", cleaned)

        # JSON integer scalar
        self.assertEqual(_sanitize_json_scalar("age", 68, plan, stats, "", False), expected_band)

        # FHIR R4 Bundle: Patient.gender="unknown", no birthDate, stripped Organization.district
        bundle = {
            "resourceType": "Bundle",
            "type": "document",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "gender": "female",
                        "birthDate": "1958-04-12",
                        "text": {"status": "generated", "div": "<div>X</div>"},
                    }
                },
                {
                    "resource": {
                        "resourceType": "Organization",
                        "name": "Apex Specialty Hospital",
                        "address": [{"city": "Bengaluru", "district": "Bengaluru Urban"}],
                    }
                },
            ],
        }
        san_b = _sanitize_fhir_bundle(bundle, plan, stats, "apex")
        pt_res = san_b["entry"][0]["resource"]
        org_res = san_b["entry"][1]["resource"]
        self.assertEqual(pt_res["gender"], "unknown")
        self.assertNotIn("birthDate", pt_res)
        self.assertIn(f"{expected_band}, unknown", pt_res["text"]["div"])
        self.assertEqual(org_res["name"], "Apex Specialty Hospital")
        self.assertNotIn("district", org_res["address"][0])
        self.assertNotIn("city", org_res["address"][0])

    def test_dob_truncated_to_yyyy_on_standard_tier(self) -> None:
        from rakshak.contracts import PrivacyPlan
        from rakshak.privacy import apply_privacy_to_fhir_bundle, apply_privacy_to_structured_record
        from rakshak.schema_profiler import ROLE_DOB

        std_plan = PrivacyPlan(
            unit_id="s::doc",
            cluster_key="ID|std",
            pseudonym="PSN-STD00001",
            date_shift_days=45,
            age_band="45-49",
            keep_district=True,
            keep_pin_prefix=True,
            generalization_level="standard",
        )
        stats = DetectionStats()
        # Structured CSV/JSON DOB -> YYYY only (not shifted date!)
        self.assertEqual(sanitize_structured_cell("19/08/1979", ROLE_DOB, "dmy", std_plan, stats), "1979")
        # Free-text DOB line -> YYYY while admission/discharge dates shift normally
        txt = "DOB: 19/08/1979\nDate of Admission: 01/03/2025\nDate of Discharge: 06/03/2025\n"
        cleaned_txt = clean_free_text(txt, stats, plan=std_plan, preferred_order="dmy")
        self.assertIn("DOB: 1979", cleaned_txt)
        self.assertNotIn("19/08/1979", cleaned_txt)
        self.assertNotIn("1979-08", cleaned_txt)
        self.assertNotIn("01/03/2025", cleaned_txt)

        # privacy.py direct helpers truncate standard DOB to YYYY too
        txt_p, changed_p = apply_privacy_to_text("DOB: 19/08/1979\nVisit: 01/03/2025\n", std_plan, "dmy")
        self.assertTrue(changed_p)
        self.assertIn("DOB: 1979", txt_p)
        rec_p, _ = apply_privacy_to_structured_record({"dob": "1979-08-19"}, std_plan, "f", {}, PolicyConfig())
        self.assertEqual(rec_p["dob"], "1979")

        # FHIR standard tier truncates Patient.birthDate to YYYY
        bundle_std = {
            "resourceType": "Bundle",
            "entry": [{"resource": {"resourceType": "Patient", "birthDate": "1979-08-19", "gender": "male"}}],
        }
        san_b = _sanitize_fhir_bundle(bundle_std, std_plan, stats, "apex")
        self.assertEqual(san_b["entry"][0]["resource"]["birthDate"], "1979")
        fb_std, _ = apply_privacy_to_fhir_bundle(
            {"resourceType": "Bundle", "entry": [{"resource": {"resourceType": "Patient", "birthDate": "1979-08-19"}}]},
            std_plan,
            PolicyConfig(),
        )
        self.assertEqual(fb_std["entry"][0]["resource"]["birthDate"], "1979")

    def test_post_widening_joint_k_and_l_diversity(self) -> None:
        # 5 identical-QI patients with condition A82.9 (k=5, l=1) + 5 patients aged 50-54 in same district with E11.9
        # Joint k-anonymity + l-diversity must widen homogeneous l=1 cells into a shared >=2-condition class
        policy = PolicyConfig(age_band_years=5, min_group_size=5, rare_condition_threshold=2, adaptive_widening=True)
        clues: list[IdentityClues] = []
        cmap: dict[str, str] = {}
        for i in range(5):
            uid = f"g1_{i}::doc"
            clues.append(IdentityClues(uid, f"g1_{i}", "doc", age=47, gender="M", district_norm="lckn", conditions=("J18.9",)))
            cmap[uid] = f"ID|g1_{i}"
        for i in range(5):
            uid = f"g2_{i}::doc"
            clues.append(IdentityClues(uid, f"g2_{i}", "doc", age=52, gender="M", district_norm="lckn", conditions=("E11.9",)))
            cmap[uid] = f"ID|g2_{i}"

        plans = build_privacy_plans(clues, cmap, policy, b"sec-kl")
        # Verify every released joint QI class has k >= 5 AND >= 2 distinct conditions
        from collections import defaultdict
        by_sig: dict[tuple, list[IdentityClues]] = defaultdict(list)
        for c in clues:
            pl = plans[c.unit_id]
            sig = (pl.age_band, c.gender if pl.keep_gender else "U", c.district_norm if pl.keep_district else "*")
            by_sig[sig].append(c)
        for sig, members in by_sig.items():
            self.assertGreaterEqual(len(members), 5, f"Joint class {sig} violated k>=5")
            distinct_conds = {cond for m in members for cond in m.conditions}
            self.assertGreaterEqual(len(distinct_conds), 2, f"Joint class {sig} failed l>=2 diversity")


if __name__ == "__main__":
    unittest.main()
