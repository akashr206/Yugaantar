"""Unseen-format ("Chaos") and meaning-trap test suite. [Person 1]"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from main import run_pipeline
from rakshak.contracts import PolicyConfig
from rakshak.detector import DetectionStats, clean_free_text
from rakshak.ingest import QuarantineError, ingest_file, sanitize_rel_path

KIT_ROOT = Path(__file__).resolve().parents[2] / "Secret-Wards-ToolKit"
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


if __name__ == "__main__":
    unittest.main()
