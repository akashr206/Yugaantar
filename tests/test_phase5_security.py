"""Phase 5 unit tests for AES-GCM at-rest vault, secret hygiene, and manifest assembly."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rakshak.security import (
    EphemeralVault,
    assert_no_secret_in_bytes,
    safe_error_reason,
    sha256_hex,
    write_release_manifests,
)


class TestPhase5Security(unittest.TestCase):
    def test_aes_gcm_quarantine_roundtrip_and_aad(self) -> None:
        vault = EphemeralVault()
        raw = b"Sensitive patient PDF bytes: Mahesh Chauhan 9876543210"
        sha = sha256_hex(raw)
        with tempfile.TemporaryDirectory() as td:
            qdir = Path(td) / "quarantine"
            enc_path = vault.quarantine_file(qdir, sha, raw)
            self.assertEqual(enc_path.name, f"{sha}.enc")
            blob = enc_path.read_bytes()
            self.assertNotIn(b"Mahesh", blob)
            self.assertEqual(vault.unseal(blob, aad=sha.encode("ascii")), raw)
            with self.assertRaises(Exception):
                vault.unseal(blob, aad=b"wrong-sha")

    def test_safe_error_reason_strips_pii(self) -> None:
        self.assertEqual(safe_error_reason(ValueError("empty_file")), "empty_file")
        self.assertEqual(
            safe_error_reason(ValueError("bad value for Ramesh 9876543210")),
            "ValueError",
        )

    def test_manifest_filters_unexported_and_blocks_secret(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            files = [
                {"input_sha256": "aaa", "output": "sanitized/a.csv", "status": "sanitized"},
                {"input_sha256": "bbb", "output": None, "status": "quarantined", "reason": "empty_file"},
            ]
            recs = [
                {"file": "sanitized/a.csv", "loc": "row:0", "patient": "PSN-123456"},
                {"file": "sanitized/b.csv", "loc": "doc", "patient": "PSN-999999"},
            ]
            rep = write_release_manifests(
                out_dir=out,
                file_entries=files,
                records_out=recs,
                identifiers_detected=5,
                records_generalized=1,
                records_suppressed=0,
                records_quarantined=0,
                elapsed_seconds=0.42,
                secret=b"my-top-secret",
            )
            self.assertEqual(rep["counts"]["files_total"], 2)
            self.assertEqual(rep["counts"]["files_sanitized"], 1)
            self.assertEqual(rep["counts"]["files_quarantined"], 1)
            self.assertEqual(rep["counts"]["records_processed"], 1)
            lines = (out / "records.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["file"], "sanitized/a.csv")

            with self.assertRaises(RuntimeError):
                assert_no_secret_in_bytes(b"my-top-secret", b"leaked my-top-secret here")


if __name__ == "__main__":
    unittest.main()
