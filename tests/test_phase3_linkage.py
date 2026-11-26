"""Phase 3 unit tests for precision-first record linkage and anti-transitive guards."""
from __future__ import annotations

import unittest

from rakshak.contracts import DateMention, IdentityClues, NameClue
from rakshak.linkage import link_records


class TestPhase3Linkage(unittest.TestCase):
    def test_homonyms_stay_separate(self) -> None:
        c1 = IdentityClues(
            unit_id="a::row:0", file_rel="a", loc="row:0",
            patient_names=(NameClue("Brijesh Yadav", ("brijesh", "yadav"), ("brj", "ytf")),),
            relative_names=(NameClue("Raju Gupta", ("raju", "gupta"), ("rj", "kpt"), "relative"),),
            mrns=("jjh0076864",), gender="M", age=70, village_norm="jlpr",
        )
        c2 = IdentityClues(
            unit_id="b::row:0", file_rel="b", loc="row:0",
            patient_names=(NameClue("Brijesh Yadav", ("brijesh", "yadav"), ("brj", "ytf")),),
            relative_names=(NameClue("Manoj Tiwari", ("manoj", "tiwari"), ("mnj", "twr"), "relative"),),
            mrns=("jjh0077243",), gender="M", age=42, village_norm="bmpr",
        )
        cmap = link_records([c1, c2])
        self.assertIsNotNone(cmap["a::row:0"])
        self.assertIsNotNone(cmap["b::row:0"])
        self.assertNotEqual(cmap["a::row:0"], cmap["b::row:0"])

    def test_fragmented_cross_script_and_dob_age_drift(self) -> None:
        reg = IdentityClues(
            unit_id="reg::row:0", file_rel="reg", loc="row:0",
            patient_names=(NameClue("आशा राजपूत", ("asha", "rajput"), ("as", "rjpt")),),
            relative_names=(NameClue("कमल राजपूत", ("kamal", "rajput"), ("kml", "rjpt"), "relative"),),
            mrns=("jjh0076900",), dob="1977-09-02", age=46, gender="F", village_norm="sltnpr",
            dates=(DateMention("16-01-2024", "2024-01-16", "clinical"),),
        )
        frag = IdentityClues(
            unit_id="note::doc", file_rel="note", loc="doc",
            patient_names=(NameClue("Asha", ("asha",), ("as",)),),
            relative_names=(NameClue("Kamal", ("kamal",), ("kml",), "relative"),),
        )
        rx = IdentityClues(
            unit_id="rx::doc", file_rel="rx", loc="doc",
            patient_names=(NameClue("आशा राजपूत", ("asha", "rajput"), ("as", "rjpt")),),
            age=48, village_norm="sltnpr",
            dates=(DateMention("10-05-2026", "2026-05-10", "clinical"),),
        )
        cmap = link_records([reg, frag, rx])
        self.assertEqual(cmap["reg::row:0"], cmap["note::doc"])
        self.assertEqual(cmap["reg::row:0"], cmap["rx::doc"])

    def test_generic_name_without_corroboration_abstains(self) -> None:
        c1 = IdentityClues(
            unit_id="n1::doc", file_rel="n1", loc="doc",
            patient_names=(NameClue("Devi Kumari", ("devi", "kumari"), ("tf", "kmr")),),
        )
        c2 = IdentityClues(
            unit_id="n2::doc", file_rel="n2", loc="doc",
            patient_names=(NameClue("Devi Kumari", ("devi", "kumari"), ("tf", "kmr")),),
        )
        cmap = link_records([c1, c2])
        self.assertIsNone(cmap["n1::doc"])
        self.assertIsNone(cmap["n2::doc"])


if __name__ == "__main__":
    unittest.main()
