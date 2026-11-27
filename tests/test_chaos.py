"""Unseen-format ("Chaos") and meaning-trap test suite."""
from __future__ import annotations

import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

from main import run_pipeline
from rakshak.contracts import PolicyConfig
from rakshak.detector import DetectionStats, clean_free_text
from rakshak.ingest import QuarantineError, ingest_file, sanitize_rel_path

_REPO_PARENT = Path(__file__).resolve().parents[2]
_DIRECT_DEV_INPUT = _REPO_PARENT / "dev" / "input"
KIT_ROOT = _REPO_PARENT if _DIRECT_DEV_INPUT.is_dir() else _REPO_PARENT / "Secret-Wards-ToolKit"
DEV_INPUT = KIT_ROOT / "dev" / "input"
SECRET = b"chaos-test-secret-key"


class TestIngestionAndTraps(unittest.TestCase):
    def test_quarantines_corrupt_mislabeled_and_deep_files(self) -> None:
        if not DEV_INPUT.is_dir():
            self.skipTest("dev/input not found")
        with self.assertRaises(QuarantineError):
            ingest_file(DEV_INPUT / "misc" / "corrupt_bytes_note.txt", DEV_INPUT, SECRET)
        with self.assertRaises(QuarantineError):
            ingest_file(DEV_INPUT / "misc" / "mislabeled_export.csv", DEV_INPUT, SECRET)
        with self.assertRaises(QuarantineError):
            ingest_file(DEV_INPUT / "misc" / "deep_export.json", DEV_INPUT, SECRET)
        with self.assertRaises(QuarantineError):
            ingest_file(DEV_INPUT / "misc" / "truncated_opd.json", DEV_INPUT, SECRET)
        with self.assertRaises(QuarantineError):
            ingest_file(DEV_INPUT / "misc" / "empty_export.csv", DEV_INPUT, SECRET)

    def test_accepts_valid_cp1252_and_utf16(self) -> None:
        if not DEV_INPUT.is_dir():
            self.skipTest("dev/input not found")
        res_cp = ingest_file(DEV_INPUT / "registration" / "arogya_legacy_export.csv", DEV_INPUT, SECRET)
        self.assertEqual(res_cp.format_type, "csv")
        res_u16 = ingest_file(DEV_INPUT / "registration" / "jeevanjyoti_patients.tsv", DEV_INPUT, SECRET)
        self.assertEqual(res_u16.format_type, "tsv")

    def test_filename_pii_sanitization(self) -> None:
        clean_1, renamed_1 = sanitize_rel_path("lab/7148988804_report_0047.json", SECRET)
        self.assertTrue(renamed_1)
        self.assertNotIn("7148988804", clean_1)

        clean_2, renamed_2 = sanitize_rel_path("discharge/Discharge_Summary_Fatima_Bano_SNJ-26-064508.txt", SECRET)
        self.assertTrue(renamed_2)
        self.assertNotIn("Fatima", clean_2)
        self.assertNotIn("064508", clean_2)

        clean_3, renamed_3 = sanitize_rel_path("clean/dengue_management_protocol.txt", SECRET)
        self.assertFalse(renamed_3)
        self.assertEqual(clean_3, "clean/dengue_management_protocol.txt")

        # Unseen hospital ID prefixes, Indic digits, Devanagari, URL-encoded, and base64 segments
        clean_4, renamed_4 = sanitize_rel_path("notes/KPR-99-887766_summary.txt", SECRET)
        self.assertTrue(renamed_4)
        self.assertNotIn("887766", clean_4)

        clean_5, renamed_5 = sanitize_rel_path("lab/९८७६५४३२१०_report.json", SECRET)
        self.assertTrue(renamed_5)
        self.assertNotIn("९८७६५४३२१०", clean_5)

        clean_6, renamed_6 = sanitize_rel_path("notes/महेश_चौहान.txt", SECRET)
        self.assertTrue(renamed_6)

        import base64
        b64_name = base64.urlsafe_b64encode(b"Fatima_Bano_9876543210").decode("ascii").rstrip("=")
        clean_7, renamed_7 = sanitize_rel_path(f"notes/{b64_name}.txt", SECRET)
        self.assertTrue(renamed_7)
        self.assertNotIn(b64_name, clean_7)

    def test_lowercase_filename_pii_caught_via_pass1_clues(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inp = root / "in"
            out = root / "out"
            (inp / "notes").mkdir(parents=True)
            (inp / "notes" / "john_doe_followup.txt").write_text(
                "Mercy Memorial Hospital\n"
                "Patient Name: John Doe  Age/Sex: 42 Yrs/M\n"
                "Diagnosis: Viral fever.\n",
                encoding="utf-8",
            )
            (inp / "notes" / "anita_summary.txt").write_text(
                "Mercy Memorial Hospital\n"
                "Patient Name: Anita Verma  Age/Sex: 35 Yrs/F\n"
                "Diagnosis: Migraine.\n",
                encoding="utf-8",
            )
            run_pipeline(inp, out, PolicyConfig(), SECRET)
            exported_names = [p.name for p in (out / "sanitized").rglob("*") if p.is_file()]
            self.assertFalse(any("john" in n.lower() or "doe" in n.lower() for n in exported_names))
            self.assertFalse(any("anita" in n.lower() for n in exported_names))

    def test_meaning_traps_preserved_and_pii_redacted(self) -> None:
        sample = (
            "Patient Name: Ramesh Yadav        Age/Sex: 65 Yrs/M\n"
            "S/o Mahesh Yadav\n"
            "Mobile: ८९२१६७८५४१\n"
            "Presenting complaints: C/O fever for 4 days.\n"
            "History: K/C/O addison's disease. H/O TB 8 yrs back.\n"
            "C/O Mahesh Yadav, H.No. 133, Ward 8\n"
            "Ward: Ward 4 (Medicine)\n"
            "Rx: Tab Ramipril 5 mg 0-0-1 x 30 days; Dolo 650 BD x 5 days\n"
            "BP 120/83, RBS 245 mg/dL (70-140).\n"
            "Dr. Ramesh Srivastava reviewed the patient.\n"
        )
        stats = DetectionStats()
        cleaned = clean_free_text(sample, stats)
        # Must remove patient/relative/phone/house
        self.assertNotIn("Ramesh Yadav", cleaned)
        self.assertNotIn("Mahesh Yadav", cleaned)
        self.assertNotIn("८९२१६७८५४१", cleaned)
        self.assertNotIn("133", cleaned)
        # Must preserve clinical traps
        self.assertIn("Dr. Ramesh Srivastava", cleaned)
        self.assertIn("C/O fever for 4 days", cleaned)
        self.assertIn("K/C/O addison's disease", cleaned)
        self.assertIn("H/O TB 8 yrs back", cleaned)
        self.assertIn("Ward 4 (Medicine)", cleaned)
        self.assertIn("Ramipril 5 mg", cleaned)
        self.assertIn("0-0-1", cleaned)
        self.assertIn("Dolo 650 BD x 5 days", cleaned)
        self.assertIn("245 mg/dL", cleaned)
        self.assertIn("70-140", cleaned)

    def test_street_names_and_prefix_honorifics_with_negative_guards(self) -> None:
        sample = (
            "Patient: Sri Ramesh Verma\n"
            "Wife: Smt. Sunita Devi, श्री कमलेश कुमार\n"
            "Address: H.No. 12, Lane MG Road, Sector 14, Distt. Kanpur\n"
            "Facility: Sanjeevani Multispeciality Hospital, Sri Venkateswara Hospital\n"
            "Doctor: Dr. Ramesh Srivastava reviewed Road traffic accident history.\n"
        )
        stats = DetectionStats()
        cleaned = clean_free_text(sample, stats)
        self.assertNotIn("Ramesh Verma", cleaned)
        self.assertNotIn("Sunita Devi", cleaned)
        self.assertNotIn("कमलेश", cleaned)
        self.assertNotIn("MG Road", cleaned)
        self.assertNotIn("Sector 14", cleaned)
        # Negative guards: clinical terms, doctors, and hospitals survive
        self.assertIn("Dr. Ramesh Srivastava", cleaned)
        self.assertIn("Sanjeevani Multispeciality Hospital", cleaned)
        self.assertIn("Sri Venkateswara Hospital", cleaned)
        self.assertIn("Road traffic accident", cleaned)

    def test_strict_csv_quarantines_malformed_quotes_but_keeps_multiline_and_ragged(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td)
            (inp / "bad_unterminated.csv").write_text('a,b\n"unterminated,123\n', encoding="utf-8")
            (inp / "bad_stray_quote.csv").write_text('a,b\nfoo"bar,123\n', encoding="utf-8")
            (inp / "valid_multiline_ragged.csv").write_text(
                'id,name,notes\n1,Asha,"Line 1\nLine 2 ""quoted"""\n2,Ramesh\n', encoding="utf-8"
            )
            with self.assertRaises(QuarantineError):
                ingest_file(inp / "bad_unterminated.csv", inp, SECRET)
            with self.assertRaises(QuarantineError):
                ingest_file(inp / "bad_stray_quote.csv", inp, SECRET)
            ok_res = ingest_file(inp / "valid_multiline_ragged.csv", inp, SECRET)
            self.assertEqual(ok_res.format_type, "csv")
            self.assertEqual(len(ok_res.parsed[2]), 2)


class TestChaosMutatedDataset(unittest.TestCase):
    def test_mutated_unseen_headers_keys_and_canaries(self) -> None:
        if not DEV_INPUT.is_dir():
            self.skipTest("dev/input not found")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            inp = tmp_path / "input"
            out = tmp_path / "output"
            shutil.copytree(DEV_INPUT, inp)

            # Mutate CSV headers to unseen column names + dot dates
            reg_csv = inp / "registration" / "sanjeevani_registration.csv"
            txt = reg_csv.read_text(encoding="utf-8-sig")
            txt = txt.replace("Patient Name", "Beneficiary_Full_Name")
            txt = txt.replace("Mobile", "Contact_Tel")
            txt = txt.replace("Aadhaar No", "UID_Twelve")
            txt = txt.replace("Village/Mohalla", "Gram_Place")
            reg_csv.write_text(txt, encoding="utf-8")

            # Inject synthetic unseen canary note
            (inp / "notes" / "unseen_canary_KPR-99-887766.txt").write_text(
                "OPD note for Maitreyi Deshpande\n"
                "UHID: KPR/99/887766\n"
                "Mobile: +91 98112 23344\n"
                "C/O cough for 2 days. Dr. Arvind Joshi advised Tab Paracetamol 500 mg 1-0-1.\n",
                encoding="utf-8",
            )

            # Mutate an OPD JSON metadata key
            opd_file = inp / "opd" / "arogya_opd_2026_02.json"
            opd_data = json.loads(opd_file.read_text(encoding="utf-8-sig"))
            opd_data["_audit_trail"] = opd_data.pop("_export_meta")
            opd_file.write_text(json.dumps(opd_data, ensure_ascii=False, indent=2), encoding="utf-8")

            policy = PolicyConfig.from_yaml(Path(__file__).resolve().parents[1] / "config" / "policy.yaml")
            report = run_pipeline(inp, out, policy, SECRET)

            self.assertEqual(report["counts"]["files_total"], 201)
            all_sanitized = "\n".join(
                p.read_text(encoding="utf-8", errors="ignore")
                for p in (out / "sanitized").rglob("*")
                if p.is_file()
            )
            # Unseen canary & filename must not leak
            self.assertNotIn("Maitreyi Deshpande", all_sanitized)
            self.assertNotIn("98112 23344", all_sanitized)
            self.assertNotIn("KPR/99/887766", all_sanitized)
            self.assertFalse(any("887766" in p.name for p in (out / "sanitized").rglob("*")))
            # Clinical terms on unseen note survive
            self.assertIn("Dr. Arvind Joshi", all_sanitized)
            self.assertIn("Paracetamol 500 mg", all_sanitized)
            self.assertIn("1-0-1", all_sanitized)

    def test_unseen_domain_generalization_crf_and_fhir_resources(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "inp"
            out = Path(td) / "out"
            inp.mkdir()

            # 1. Unanchored prose with unseen Indian hospitals, drugs, ICD-10 U07.1, package SE002B, @okaxis, Hindi PIN/district
            (inp / "note_south.txt").write_text(
                "Apollo Specialty Hospital, Bengaluru\n"
                "Karan Malhotra presented with high fever on September 5, 2026.\n"
                "DOB: 15/06/12\n"
                "ABHA: 23-4455-6677-8899 | UPI: karan.m@okaxis | Reg. No.: KGMU/26/00123\n"
                "Crocin 500 BD x 5 days; Diagnosis: COVID-19 (U07.1); Package SE002B\n"
                "जिला: वाराणसी, पिन कोड: २२१००१\n"
                "Dr. Meenakshi Iyer noted Parkinson's disease history.\n",
                encoding="utf-8",
            )
            # 2. Multi-ticket JSON array + arbitrary source_path ending in .csv + Blood Pressure TitleCase key
            (inp / "helpdesk.json").write_text(
                json.dumps(
                    {
                        "_export_meta": {"source_path": r"D:\HIS\Rohan_Malhotra\2026\opd\batch.csv"},
                        "panels": {"Blood Pressure": "120/80"},
                        "tickets": [
                            {"id": "T1", "patient_name": "Priya Nair", "mobile": "9811223344", "note": "bill query"},
                            {"id": "T2", "patient_name": "Deepak Bansal", "mobile": "9822334455", "note": "report issue"},
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            # 3. FHIR R4 bundle with Observation.subject.display and RelatedPerson
            (inp / "fhir_ext.json").write_text(
                json.dumps(
                    {
                        "resourceType": "Bundle",
                        "type": "collection",
                        "entry": [
                            {
                                "resource": {
                                    "resourceType": "Patient",
                                    "name": [{"text": "Ananya Deshmukh"}],
                                    "gender": "female",
                                    "birthDate": "1990-05-10",
                                }
                            },
                            {
                                "resource": {
                                    "resourceType": "RelatedPerson",
                                    "name": [{"text": "Vikram Deshmukh"}],
                                    "telecom": [{"system": "phone", "value": "9876543210"}],
                                }
                            },
                            {
                                "resource": {
                                    "resourceType": "Observation",
                                    "status": "final",
                                    "code": {"text": "Blood Glucose"},
                                    "subject": {"reference": "urn:uuid:11111111-1111-1111-1111-111111111111", "display": "Ananya Deshmukh"},
                                    "valueQuantity": {"value": 118, "unit": "mg/dL"},
                                }
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            policy = PolicyConfig.from_yaml(Path(__file__).resolve().parents[1] / "config" / "policy.yaml")
            run_pipeline(inp, out, policy, SECRET)

            note_out = (out / "sanitized" / "note_south.txt").read_text(encoding="utf-8")
            self.assertNotIn("Karan Malhotra", note_out)
            self.assertNotIn("23-4455-6677-8899", note_out)
            self.assertNotIn("karan.m@okaxis", note_out)
            self.assertNotIn("KGMU/26/00123", note_out)
            self.assertNotIn("२२१००१", note_out)
            self.assertNotIn("वाराणसी", note_out)
            self.assertNotIn("15/06/12", note_out)
            self.assertIn("Apollo Specialty Hospital", note_out)
            self.assertIn("Crocin 500 BD x 5 days", note_out)
            self.assertIn("U07.1", note_out)
            self.assertIn("SE002B", note_out)
            self.assertIn("Dr. Meenakshi Iyer", note_out)
            self.assertIn("Parkinson's disease", note_out)

            hd_out = json.loads((out / "sanitized" / "helpdesk.json").read_text(encoding="utf-8"))
            self.assertNotIn("Rohan_Malhotra", hd_out["_export_meta"]["source_path"])
            self.assertIn("Blood Pressure", hd_out["panels"])
            self.assertEqual(len(hd_out["tickets"]), 2)
            self.assertNotEqual(hd_out["tickets"][0]["patient_pseudonym"], hd_out["tickets"][1]["patient_pseudonym"])

            fh_out = json.loads((out / "sanitized" / "fhir_ext.json").read_text(encoding="utf-8"))
            raw_fh = json.dumps(fh_out)
            self.assertNotIn("Ananya Deshmukh", raw_fh)
            self.assertNotIn("Vikram Deshmukh", raw_fh)
            self.assertNotIn("9876543210", raw_fh)
            obs_subj = fh_out["entry"][2]["resource"]["subject"]["display"]
            self.assertTrue(obs_subj.startswith("PSN-") or obs_subj == "[PATIENT_NAME]")

    def test_unseen_opaque_columns_json_labels_ids_and_dates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "inp"
            out = Path(td) / "out"
            inp.mkdir()

            # 1. Completely unseen opaque CSV headers (zero semantic header hints!)
            (inp / "opaque_cohort.csv").write_text(
                "col_a,col_b,col_c,col_d,col_e,col_f,col_g,obs_val\n"
                "ZLH-2026-0091,Anirudh,42,M,13-Jan-1984,987-654-3210,27481920384712,13.4\n"
                "ZLH_26_0092,Meera Nair,67,Female,5th May 1959,(+91) 98234.56789,34125678901234,5.6\n"
                "ZLH-2026-0093,Vikramjit Singh,29,M,19-Oct-1997,0091 98111 22233,45123456789012,7.4\n",
                encoding="utf-8",
            )

            # 2. Unseen opaque JSON keys linking via phone/MRN to opaque CSV row 1 ("Meera Nair")
            (inp / "unseen_events.json").write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "z1": "ZLH_26_0092",
                                "z2": "Meera Nair",
                                "z3": 67,
                                "z4": "F",
                                "z5": "1959-05-05",
                                "z6": "98234-56789",
                                "z7": "Serum Creatinine 1.1 mg/dL | Dolo 650 BD x 5 days",
                            },
                            {
                                "z1": "ZLH-2026-0093",
                                "z2": "Vikramjit Singh",
                                "z3": 29,
                                "z4": "M",
                                "z5": "1997-10-19",
                                "z6": "9811122233",
                                "z7": "Haemoglobin 12.1 g/dL | Tab Amlodipine 5 mg 1-0-1",
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            # 3. Unseen free-text field labels, delimiters (=, -), ID shapes, UPIs, dates, plus clinical false-positive traps
            (inp / "unseen_note.txt").write_text(
                "Subject = Arjun Subramanian\n"
                "Caregiver - Devika Subramanian\n"
                "Emergency Contact = (+91) 981-234-5678\n"
                "Chart Ref # KGMU_26_7712 | Episode No = P-998811 | Year Ref: 2026/CMC/00412\n"
                "ABHA: 27481920384712 | UPI: arjun.s@federal | Aadhaar: (1234).5678.9012 | Tel: 0522-2612345\n"
                "Birth Dt: 19-Oct-1998\n"
                "Date of Admission: 13-Jan-2026\n"
                "Date of Discharge: 18-Jan-2026\n"
                "Visit on October 19 | Reviewed: 5th September 2026 | ISO-text: 2026-Jan-13\n"
                "Case Status - stable | Panel Name: Complete Blood Count | Drug Name: Ceftriaxone 1g IV\n"
                "Platelet 225000 /uL; HBV-1234 negative; Covid-19 (U07.1); 10 may indicate mild dehydration.\n",
                encoding="utf-8",
            )

            policy = PolicyConfig.from_yaml(Path(__file__).resolve().parents[1] / "config" / "policy.yaml")
            run_pipeline(inp, out, policy, SECRET)

            csv_out = (out / "sanitized" / "opaque_cohort.csv").read_text(encoding="utf-8")
            # Single-word AND multi-word unseen names, exact ages, exact DOBs, 3-3-4 phones, 14-digit ABHA redacted/banded
            for leaked in (
                "Anirudh", "Meera Nair", "Vikramjit Singh",
                "42,", "67,", "29,",
                "13-Jan-1984", "5th May 1959", "19-Oct-1997",
                "987-654-3210", "98234.56789", "98111 22233",
                "27481920384712", "34125678901234", "45123456789012",
            ):
                self.assertNotIn(leaked, csv_out)
            # Numeric clinical lab column "obs_val" preserved verbatim
            self.assertIn("13.4", csv_out)
            self.assertIn("5.6", csv_out)
            self.assertIn("7.4", csv_out)
            self.assertIn("PSN-", csv_out)

            json_out = (out / "sanitized" / "unseen_events.json").read_text(encoding="utf-8")
            for leaked in ("Meera Nair", "Vikramjit Singh", "98234-56789", "9811122233"):
                self.assertNotIn(leaked, json_out)
            self.assertIn("Serum Creatinine 1.1 mg/dL", json_out)
            self.assertIn("1-0-1", json_out)

            # Verify Pass-1 linkage connected opaque CSV + opaque JSON rows to identical PSN!
            recs = [json.loads(line) for line in (out / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            rec_map = {(r["file"], r["loc"]): r["patient"] for r in recs}
            self.assertIsNotNone(rec_map[("sanitized/opaque_cohort.csv", "row:1")])
            self.assertEqual(
                rec_map[("sanitized/opaque_cohort.csv", "row:1")],
                rec_map[("sanitized/unseen_events.json", "/entries/0")],
            )
            self.assertEqual(
                rec_map[("sanitized/opaque_cohort.csv", "row:2")],
                rec_map[("sanitized/unseen_events.json", "/entries/1")],
            )

            note_out = (out / "sanitized" / "unseen_note.txt").read_text(encoding="utf-8")
            for leaked in (
                "Arjun Subramanian", "Devika Subramanian", "981-234-5678",
                "KGMU_26_7712", "P-998811", "2026/CMC/00412",
                "27481920384712", "arjun.s@federal", "(1234).5678.9012", "0522-2612345",
                "19-Oct-1998", "13-Jan-2026", "18-Jan-2026",
                "October 19", "5th September 2026", "2026-Jan-13",
            ):
                self.assertNotIn(leaked, note_out)
            self.assertIn("[DOB]", note_out)
            # Stay interval (13-Jan-2026 -> 18-Jan-2026 = 5 days) preserved across hyphenated month dates!
            from rakshak.clue_extractor import parse_date_string as _pds
            import datetime as _dt
            adm_m = re.search(r"Date of Admission:\s*(.+)", note_out)
            dis_m = re.search(r"Date of Discharge:\s*(.+)", note_out)
            self.assertIsNotNone(adm_m)
            self.assertIsNotNone(dis_m)
            d_adm = _dt.date.fromisoformat(_pds(adm_m.group(1), "dmy") or "")
            d_dis = _dt.date.fromisoformat(_pds(dis_m.group(1), "dmy") or "")
            self.assertEqual((d_dis - d_adm).days, 5)
            # ISO-text single pass produces exact YYYY-Mon-DD format (no double-shift)
            self.assertRegex(note_out, r"ISO-text:\s*\d{4}-[A-Za-z]{3}-\d{2}")
            # Clinical meaning-trap preservation
            self.assertIn("Case Status - stable", note_out)
            self.assertIn("Panel Name: Complete Blood Count", note_out)
            self.assertIn("Ceftriaxone 1g IV", note_out)
            self.assertIn("225000 /uL", note_out)
            self.assertIn("HBV-1234 negative", note_out)
            self.assertIn("Covid-19 (U07.1)", note_out)
            self.assertIn("10 may indicate mild dehydration", note_out)

    def test_evasion_homoglyph_zw_inline_encoding_verhoeff_and_crf_multitag(self) -> None:
        import base64
        from rakshak.translit import is_valid_verhoeff

        # Find a real Verhoeff-valid 12-digit number starting with '2' dynamically for test precision
        valid_aadhaar = next(f"23412345678{d}" for d in "0123456789" if is_valid_verhoeff(f"23412345678{d}"))
        invalid_12 = next(f"23412345678{d}" for d in "0123456789" if not is_valid_verhoeff(f"23412345678{d}"))

        inner_secret = "Patient: Kavita Deshmukh, Tel: 9811223344"
        b64_blob = base64.b64encode(inner_secret.encode("utf-8")).decode("ascii")
        hex_blob = "Patient: Meera Iyer 9876501234".encode("utf-8").hex()
        # Cyrillic homoglyphs inside 'Patient' ('Р' U+0420, 'а' U+0430, 'е' U+0435) + zero-width space inside phone
        sample = (
            "Р\u200bat\u200biеnt Name: Ranvir Gill\n"
            "Sangram Das ko 4 din se bukhar hai.\n"
            "Mobile: 98\u200b76\u200b543210\n"
            f"Unlabeled ID: {valid_aadhaar} | Lot Code: {invalid_12}\n"
            f"Attachment: {b64_blob}\n"
            f"HexNote: {hex_blob}\n"
            "Encoded: Patient%20Name%3A%20Sandeep%20Kulkarni%20Tel%3A%209822334455\n"
            "Dr. Ranvir Mehta advised Tab Azithromycin 500 mg OD.\n"
        )
        stats = DetectionStats()
        cleaned = clean_free_text(sample, stats)

        # Homoglyph + zero-width + -vir/-gram names caught
        self.assertNotIn("Ranvir Gill", cleaned)
        self.assertNotIn("Sangram Das", cleaned)
        self.assertNotIn("9876543210", cleaned.replace("\u200b", ""))
        # Verhoeff bare Aadhaar caught while non-Verhoeff random 12-digit lot code stays
        self.assertNotIn(valid_aadhaar, cleaned)
        self.assertIn(invalid_12, cleaned)
        # Encoded payloads do not leak inner plaintext
        self.assertNotIn(b64_blob, cleaned)
        self.assertNotIn("Kavita Deshmukh", cleaned)
        self.assertNotIn("Meera Iyer", cleaned)
        self.assertNotIn(hex_blob, cleaned)
        self.assertNotIn("Sandeep Kulkarni", cleaned)
        # Clinical traps preserved
        self.assertIn("Dr. Ranvir Mehta", cleaned)
        self.assertIn("Azithromycin 500 mg OD", cleaned)


if __name__ == "__main__":
    unittest.main()
