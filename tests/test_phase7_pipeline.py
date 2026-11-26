"""Phase 7 unit tests for the Sync 2 Person 2 pipeline facade."""
from __future__ import annotations

import unittest

from rakshak.contracts import PolicyConfig, RecordUnit
from rakshak.pipeline_p2 import run_person2_pipeline


class TestPhase7PipelineFacade(unittest.TestCase):
    def test_run_person2_pipeline_end_to_end(self) -> None:
        units = [
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
        plans, date_orders = run_person2_pipeline(units, PolicyConfig(), b"sync2-secret")
        self.assertEqual(date_orders[("reg/a.csv", "Reg Date")], "dmy")
        p1 = plans["reg/a.csv::row:0"]
        p2 = plans["notes/n1.txt::doc"]
        self.assertIsNotNone(p1.pseudonym)
        self.assertEqual(p1.pseudonym, p2.pseudonym)
        self.assertEqual(p1.date_shift_days, p2.date_shift_days)


if __name__ == "__main__":
    unittest.main()
