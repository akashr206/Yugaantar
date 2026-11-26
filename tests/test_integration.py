"""End-to-end integration tests for `main.run_pipeline`."""
from __future__ import annotations

import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

from main import run_pipeline
from rakshak.contracts import PolicyConfig

_REPO_PARENT = Path(__file__).resolve().parents[2]
_DIRECT_DEV_INPUT = _REPO_PARENT / "dev" / "input"
KIT_ROOT = _REPO_PARENT if _DIRECT_DEV_INPUT.is_dir() else _REPO_PARENT / "Secret-Wards-ToolKit"
DEV_INPUT = KIT_ROOT / "dev" / "input"
SECRET = b"integration-test-secret-key"


class TestFullPipelineIntegration(unittest.TestCase):
    def test_synthetic_end_to_end_linkage_quarantine_and_secret_safety(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inp = root / "in"
            out = root / "out"
            inp.mkdir(parents=True)

            (inp / "reg.csv").write_text(
                "UHID,Patient Name,Age,Gender,District,PIN,Admission Date,Discharge Date\n"
                "SNJ/26/064127,Mahesh Chauhan,48,M,Rae Bareli,229073,01/03/2025,05/03/2025\n",
                encoding="utf-8",
            )
            (inp / "note.txt").write_text(
                "Patient Name: महेश चौहान\n"
                "UHID: SNJ/26/064127\n"
                "Phone: +91 9876543210\n"
                "Note: Admitted on 01/03/2025, discharged on 05/03/2025 for chest pain.\n",
                encoding="utf-8",
            )
            (inp / "bad_binary.csv").write_bytes(b"\xff\xfe\x00\x00\x80\x81\x82")

            report = run_pipeline(inp, out, PolicyConfig(), SECRET)
            self.assertEqual(report["counts"]["files_sanitized"], 2)
            self.assertEqual(report["counts"]["files_quarantined"], 1)

            quar_files = list((out / "quarantine").glob("*.enc"))
            self.assertEqual(len(quar_files), 1)
            q_bytes = quar_files[0].read_bytes()
            self.assertNotIn(SECRET, q_bytes)
            self.assertNotIn(b"Mahesh", q_bytes)

            recs = [
                json.loads(line)
                for line in (out / "records.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            psns = {r["patient"] for r in recs}
            self.assertEqual(len(psns), 1)
            psn = next(iter(psns))
            self.assertRegex(psn or "", r"^PSN-[A-Z0-9]+$")

            for p in out.rglob("*"):
                if p.is_file():
                    self.assertNotIn(SECRET, p.read_bytes(), f"secret leaked into {p}")

            san_note = (out / "sanitized" / "note.txt").read_text(encoding="utf-8")
            self.assertNotIn("9876543210", san_note)
            self.assertNotIn("महेश", san_note)
            self.assertIn("chest pain", san_note)

    def test_fhir_period_end_and_iso_day_lt20_shift(self) -> None:
        import datetime as dt

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inp = root / "in"
            out = root / "out"
            (inp / "fhir").mkdir(parents=True)
            bundle = {
                "resourceType": "Bundle",
                "type": "document",
                "meta": {"lastUpdated": "2026-02-06T09:00:00+05:30", "source": "#ADM000162"},
                "entry": [
                    {
                        "fullUrl": "urn:uuid:6cd93b57-6d6d-453c-aaf2-910234567890",
                        "resource": {
                            "resourceType": "Patient",
                            "identifier": [{"system": "https://hosp.example/uhid", "value": "ADM000162"}],
                            "name": [{"text": "Ashok Patel"}],
                            "telecom": [{"system": "phone", "value": "+91-8069773173"}],
                            "gender": "male",
                            "birthDate": "1979-04-20",
                        },
                    },
                    {
                        "fullUrl": "urn:uuid:0b164e7e-56c9-43b2-9456-8f8912345678",
                        "resource": {
                            "resourceType": "Encounter",
                            "status": "finished",
                            "subject": {"reference": "urn:uuid:6cd93b57-6d6d-453c-aaf2-910234567890"},
                            "serviceProvider": {"reference": "urn:uuid:0b164e7e-56c9-43b2-9456-8f8912345678"},
                            "period": {
                                "start": "2026-02-01T14:00:00+05:30",
                                "end": "2026-02-06T09:00:00+05:30",
                            },
                        },
                    },
                ],
            }
            (inp / "fhir" / "bundle_test.json").write_text(json.dumps(bundle), encoding="utf-8")
            (inp / "note_plus91.txt").write_text(
                "UHID: ADM000162\nTel: +91-8069773173\nRBS 70-140 mg/dL.\n", encoding="utf-8"
            )
            run_pipeline(inp, out, PolicyConfig(), b"yug-final-verify-secret")
            san_b = json.loads((out / "sanitized" / "fhir" / "bundle_test.json").read_text(encoding="utf-8"))
            pat = san_b["entry"][0]["resource"]
            self.assertNotIn("birthDate", pat)
            self.assertEqual(pat["telecom"][0]["value"], "[PHONE]")
            enc = san_b["entry"][1]["resource"]
            self.assertEqual(enc["subject"]["reference"], "urn:uuid:6cd93b57-6d6d-453c-aaf2-910234567890")
            self.assertEqual(enc["serviceProvider"]["reference"], "urn:uuid:0b164e7e-56c9-43b2-9456-8f8912345678")
            period = enc["period"]
            self.assertNotIn("RELATIVE_NAME", period["end"])
            self.assertNotEqual(period["start"][:10], "2026-02-01")
            self.assertNotEqual(period["end"][:10], "2026-02-06")
            d_start = dt.date.fromisoformat(period["start"][:10])
            d_end = dt.date.fromisoformat(period["end"][:10])
            self.assertEqual((d_end - d_start).days, 5)
            self.assertNotEqual(san_b["meta"]["lastUpdated"][:10], "2026-02-06")
            note_out = (out / "sanitized" / "note_plus91.txt").read_text(encoding="utf-8")
            self.assertNotIn("8069773173", note_out)
            self.assertIn("70-140", note_out)

    @unittest.skipUnless(DEV_INPUT.is_dir(), "dev/input not found")
    def test_dev_input_quarantine_doc_report_and_secret_safety(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            report = run_pipeline(DEV_INPUT, out, PolicyConfig(), SECRET)

            self.assertGreater(report["counts"]["files_sanitized"], 0)
            self.assertGreater(report["counts"]["files_quarantined"], 0)

            quar_files = list((out / "quarantine").glob("*.enc"))
            self.assertEqual(len(quar_files), report["counts"]["files_quarantined"])
            for qf in quar_files:
                qb = qf.read_bytes()
                self.assertGreater(len(qb), 0)
                self.assertNotIn(SECRET, qb)
                self.assertNotIn(b'"patient"', qb)

            rep_bytes = (out / "report.json").read_bytes()
            rec_bytes = (out / "records.jsonl").read_bytes()
            self.assertNotIn(SECRET, rep_bytes)
            self.assertNotIn(SECRET, rec_bytes)

            # lab/report_00005.json must map to a single doc unit with a linked PSN-XXXXXX
            target_in = DEV_INPUT / "lab" / "report_00005.json"
            if target_in.is_file():
                in_sha = hashlib.sha256(target_in.read_bytes()).hexdigest()
                matching_entries = [
                    e for e in report["files"] if e.get("input_sha256") == in_sha
                ]
                self.assertEqual(len(matching_entries), 1)
                self.assertEqual(matching_entries[0]["status"], "sanitized")
                out_file_rel = matching_entries[0]["output"]

                recs = [
                    json.loads(line)
                    for line in rec_bytes.decode("utf-8").splitlines()
                    if line.strip()
                ]
                doc_recs = [r for r in recs if r["file"] == out_file_rel]
                self.assertEqual(len(doc_recs), 1)
                self.assertEqual(doc_recs[0]["loc"], "doc")
                self.assertIsNotNone(doc_recs[0]["patient"])
                self.assertTrue(re.match(r"^PSN-[A-Z0-9]+$", doc_recs[0]["patient"]))

            for p in (out / "sanitized").rglob("*"):
                if p.is_file():
                    self.assertNotIn(SECRET, p.read_bytes())

    def test_suppressed_free_text_district_and_dob_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inp = root / "in"
            out = root / "out"
            inp.mkdir(parents=True)
            (inp / "hindi_note.txt").write_text(
                "जिला अस्पताल हरदोई\n"
                "Patient Name: रीना निषाद  Age/Sex: 65 Yrs/F\n"
                "DOB: 19/08/1960\n"
                "पता: मकान नं. 398, ग्राम भीमनगर, तहसील शाहबाद, जिला हरदोई - 241124\n"
                "Diagnosis: Rabies, unspecified (A82.9)\n",
                encoding="utf-8",
            )
            run_pipeline(inp, out, PolicyConfig(), b"sec-hardening-test")
            cleaned = (out / "sanitized" / "hindi_note.txt").read_text(encoding="utf-8")
            self.assertIn("जिला अस्पताल हरदोई", cleaned)
            self.assertNotIn("19/08/1960", cleaned)
            self.assertIn("[DOB]", cleaned)
            self.assertIn("[DISTRICT]", cleaned)
            self.assertNotIn("जिला हरदोई", cleaned)


if __name__ == "__main__":
    unittest.main()
