"""Phase 1 unit tests for cross-script transliteration, phonetic normalization, and ID canonicalization."""
from __future__ import annotations

import unittest

from rakshak.translit import (
    consonant_skeleton,
    devanagari_to_latin,
    extract_name_tokens,
    name_tokens_match,
    norm_aadhaar,
    norm_aadhaar_last4,
    norm_abha,
    norm_district,
    norm_location,
    norm_mrn,
    norm_phone,
)


class TestPhase1Translit(unittest.TestCase):
    def test_cross_script_names_match(self) -> None:
        pairs = [
            ("आशा राजपूत", "Asha Rajput"),
            ("नरेश चौधरी", "Naresh Choudhary"),
            ("मोहम्मद शाहिद", "Mohd. Shahid"),
            ("मोहम्मद शाहिद", "Mohammed Shahid"),
            ("पार्वती पांडे", "Parvathi Pandeya"),
            ("सुनीता पटेल", "Suneeta Patel"),
            ("रीना निषाद", "Reena Nishad"),
            ("ज़ैनब सिद्दीकी", "Zainab Siddiqui"),
            ("फ़ातिमा ख़ातून", "Fatima Khatoon"),
            ("अकरम शेख़", "Akram Shaikh"),
            ("ज़ुबैर क़ुरैशी", "Zubair Quraishi"),
            ("बृजेश यादव", "BRIJESH YADAV"),
            ("Aneeta Sarma", "अनीता शर्मा"),
            ("Vikash Kumar", "Vikas Kumar"),
            ("Devi Kumari", "देवी कुमारी"),
        ]
        for a, b in pairs:
            ta = extract_name_tokens(a)
            tb = extract_name_tokens(b)
            score = name_tokens_match(ta, tb)
            self.assertGreaterEqual(
                score, 0.85, f"Expected match for {a!r} ({ta}) vs {b!r} ({tb}), got {score}"
            )

    def test_honorifics_and_order_variants(self) -> None:
        self.assertEqual(extract_name_tokens("Asha ji"), ("asha",))
        self.assertEqual(extract_name_tokens("Shahid sahab"), ("shahid",))
        self.assertEqual(extract_name_tokens("Sunita amma"), ("sunita",))
        self.assertGreaterEqual(
            name_tokens_match(
                extract_name_tokens("Shahid sahab"),
                extract_name_tokens("Mohammad Shahid"),
            ),
            0.75,
        )
        self.assertGreaterEqual(
            name_tokens_match(
                extract_name_tokens("Shahid Mohammad"),
                extract_name_tokens("Mohammad Shahid"),
            ),
            0.90,
        )

    def test_distinct_names_do_not_collide(self) -> None:
        non_pairs = [
            ("Asha Rajput", "Usha Tiwari"),
            ("Anil Chaudhary", "Sunil Chaudhary"),
            ("Kamal Thakur", "Kamla Devi"),
            ("Ramesh Patel", "Mahesh Patel"),
            ("Devi", "Kamla Devi"),
            ("Devi Kumari", "Kamla Devi"),
            ("Sunita Kumari", "Sunita Devi"),
        ]
        for a, b in non_pairs:
            score = name_tokens_match(extract_name_tokens(a), extract_name_tokens(b))
            self.assertLess(
                score, 0.75, f"Unexpected collision for {a!r} vs {b!r}: {score}"
            )

    def test_locations_and_districts_normalize(self) -> None:
        self.assertEqual(
            consonant_skeleton(norm_location("मोहनगंज कलां").split()[0]),
            consonant_skeleton(norm_location("Mohanganj Kalan").split()[0]),
        )
        self.assertEqual(norm_location("भीमपुर"), norm_location("Bhimpur"))
        self.assertEqual(norm_district("Rae Bareli"), norm_district("raebareli"))
        self.assertEqual(norm_district("बाराबंकी"), norm_district("Barabanki"))
        self.assertEqual(norm_location("रसूलखेड़ा"), norm_location("Rasoolkheda"))
        self.assertEqual(norm_location("शिवगढ़"), norm_location("Shivgarh"))
        self.assertEqual(devanagari_to_latin("रसूलखेड़ा"), "rasulakheda")

    def test_id_normalizers(self) -> None:
        self.assertEqual(norm_phone("+91 72864 90658"), "7286490658")
        self.assertEqual(norm_phone("060062-72163"), "6006272163")
        self.assertEqual(norm_phone("७००८६५३४२२"), "7008653422")
        self.assertIsNone(norm_phone("1234567890"))

        self.assertEqual(norm_aadhaar("8799 6191 4045"), "879961914045")
        self.assertIsNone(norm_aadhaar("XXXX XXXX 5866"))
        self.assertEqual(norm_aadhaar_last4("XXXX XXXX 5866"), "5866")

        self.assertEqual(norm_abha("91-7938-3729-8403"), "91793837298403")

        self.assertEqual(norm_mrn("SNJ/26/064127"), ("snj26064127", "064127", "snj"))
        self.assertEqual(norm_mrn("JJH-0076864"), ("jjh0076864", "0076864", "jjh"))
        self.assertEqual(norm_mrn("UHID-010607"), ("uhid010607", "010607", "ass"))
        self.assertEqual(norm_mrn("DHH2026-084999"), ("dhh2026084999", "084999", "dhh"))


if __name__ == "__main__":
    unittest.main()
