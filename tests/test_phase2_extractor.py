"""Phase 2 unit tests for multi-format clue extraction and DMY/MDY date inference."""
from __future__ import annotations

import unittest

from rakshak.clue_extractor import extract_all_clues, infer_file_date_orders, parse_date_string
from rakshak.contracts import RecordUnit


class TestPhase2Extractor(unittest.TestCase):
    def test_dmy_vs_mdy_date_inference(self) -> None:
        units = [
            RecordUnit(
                unit_id="registration/arogya_legacy.csv::row:0",
                file_rel="registration/arogya_legacy.csv",
                loc="row:0",
                format_type="csv",
                hospital_hint="ass",
                structured_fields={"DOB (MM/DD/YYYY)": "05/30/1986", "Registered": "03/08/2026"},
            ),
            RecordUnit(
                unit_id="registration/sanjeevani.csv::row:0",
                file_rel="registration/sanjeevani.csv",
                loc="row:0",
                format_type="csv",
                hospital_hint="snj",
                structured_fields={"DOB": "20/09/1959", "Reg Date": "07/02/2025"},
            ),
        ]
        orders = infer_file_date_orders(units)
        self.assertEqual(orders[("registration/arogya_legacy.csv", "DOB (MM/DD/YYYY)")], "mdy")
        self.assertEqual(orders[("registration/sanjeevani.csv", "DOB")], "dmy")
        self.assertEqual(parse_date_string("05/30/1986", "mdy"), "1986-05-30")
        self.assertEqual(parse_date_string("07/02/2025", "dmy"), "2025-02-07")

    def test_patient_name_not_misclassified_as_relative_and_no_history_age(self) -> None:
        u = RecordUnit(
            unit_id="opd/jjh.json::/visits/0",
            file_rel="opd/jjh.json",
            loc="/visits/0",
            format_type="json",
            hospital_hint="jjh",
            structured_fields={
                "patient_ref": "JJH-0076893",
                "patient_name": "Neha Paswan",
                "history": "H/O TB 10 yrs back, completed ATT",
                "complaints": "6 दिन से पेट दर्द",
                "notes": "Neha didi ka BP 121/71",
            },
        )
        clue = extract_all_clues([u])[0]
        self.assertEqual([n.latin_tokens for n in clue.patient_names], [("neha", "paswan"), ("neha",)])
        self.assertEqual(clue.relative_names, ())
        self.assertIsNone(clue.age)
        self.assertEqual(clue.mrns, ("jjh0076893",))

    def test_abha_does_not_bleed_into_aadhaar(self) -> None:
        u = RecordUnit(
            unit_id="discharge/ds.txt::doc",
            file_rel="discharge/ds.txt",
            loc="doc",
            format_type="txt",
            hospital_hint="snj",
            raw_text=(
                "Patient Name: Sukhwinder Kaur  Age/Sex: 44 Yrs/F\n"
                "UHID: SNJ/26/064423  ABHA: 91-7529-2330-5116\n"
                "Consultant: Dr. Ramesh Srivastava\n"
            ),
        )
        clue = extract_all_clues([u])[0]
        self.assertEqual(clue.abhas, ("91752923305116",))
        self.assertEqual(clue.aadhaars, ())
        self.assertEqual(clue.age, 44)
        self.assertEqual(clue.gender, "F")
        # Doctor name must not be captured as patient name
        self.assertEqual([n.latin_tokens for n in clue.patient_names], [("sukhwinder", "kaur")])

    def test_co_complaint_not_extracted_as_relative(self) -> None:
        u = RecordUnit(
            unit_id="notes/n1.txt::doc",
            file_rel="notes/n1.txt",
            loc="doc",
            format_type="txt",
            hospital_hint="ass",
            raw_text=(
                "NURSING NOTES - IP/26/00134\n"
                "Presenting complaints: C/O fever for 9 days.\n"
                "Attendant (S/o Zubair) informed about diet.\n"
            ),
        )
        clue = extract_all_clues([u])[0]
        self.assertEqual(clue.mrns, ("ip2600134",))
        self.assertEqual([r.latin_tokens for r in clue.relative_names], [("zubair",)])

    def test_unseen_chaos_mrn_prefixes(self) -> None:
        u = RecordUnit(
            unit_id="notes/chaos.txt::doc",
            file_rel="notes/chaos.txt",
            loc="doc",
            format_type="txt",
            raw_text="Reg. No.: KGMU/26/009912\nMRN: SGPGI-882210\n",
        )
        clue = extract_all_clues([u])[0]
        self.assertIn("kgmu26009912", clue.mrns)
        self.assertIn("sgpgi882210", clue.mrns)


if __name__ == "__main__":
    unittest.main()
