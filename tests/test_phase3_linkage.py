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

    def test_abbreviation_and_initial_linkage_with_strong_id_and_conflict_guard(self) -> None:
        from rakshak.translit import extract_name_tokens, phonetic_name_tokens

        def _nc(raw: str) -> NameClue:
            return NameClue(raw, extract_name_tokens(raw), phonetic_name_tokens(raw))

        a1 = IdentityClues("a1::doc", "a1", "doc", patient_names=(_nc("Sk Tariq"),), phones=("9876501234",))
        a2 = IdentityClues("a2::doc", "a2", "doc", patient_names=(_nc("Shaikh Tariq"),), phones=("9876501234",))
        b1 = IdentityClues("b1::doc", "b1", "doc", patient_names=(_nc("R. K. Verma"),), aadhaars=("112233445566",))
        b2 = IdentityClues("b2::doc", "b2", "doc", patient_names=(_nc("Ramesh Kumar Verma"),), aadhaars=("112233445566",))
        # Weak initial without hard ID must stay unlinked (no false merge)
        c1 = IdentityClues("c1::doc", "c1", "doc", patient_names=(_nc("R. Sharma"),), village_norm="rmpr", age=40)
        c2 = IdentityClues("c2::doc", "c2", "doc", patient_names=(_nc("Rajesh Sharma"),), village_norm="rmpr", age=40)

        cmap = link_records([a1, a2, b1, b2, c1, c2])
        self.assertEqual(cmap["a1::doc"], cmap["a2::doc"])
        self.assertEqual(cmap["b1::doc"], cmap["b2::doc"])
        self.assertTrue(cmap["c1::doc"] is None or cmap["c1::doc"] != cmap["c2::doc"])

    def test_clusters_compatible_rejects_transitive_multi_name_conflict(self) -> None:
        from rakshak.linkage import _ClusterState, _clusters_compatible

        c_a = _ClusterState(unit_ids=["a1", "a2"], full_name_tokens=[("ramesh", "kumar"), ("ramesh", "verma")])
        c_b = _ClusterState(unit_ids=["b1"], full_name_tokens=[("ramesh", "kumar")])
        # Conflicting full name ("ramesh", "verma") vs ("ramesh", "kumar") -> any() would pass, all() must block
        self.assertFalse(_clusters_compatible(c_a, c_b))

    def test_corroborator4_enforces_age_within_two_years(self) -> None:
        c1 = IdentityClues(
            unit_id="a::doc", file_rel="a", loc="doc",
            patient_names=(NameClue("Meena Rawat", ("meena", "rawat"), ("mn", "rwt")),),
            aadhaar_last4=("4321",), district_norm="lckn", age=40,
        )
        c2_far = IdentityClues(
            unit_id="b::doc", file_rel="b", loc="doc",
            patient_names=(NameClue("Meena Rawat", ("meena", "rawat"), ("mn", "rwt")),),
            aadhaar_last4=("4321",), district_norm="lckn", age=43,
        )
        c3_missing_age = IdentityClues(
            unit_id="c::doc", file_rel="c", loc="doc",
            patient_names=(NameClue("Meena Rawat", ("meena", "rawat"), ("mn", "rwt")),),
            aadhaar_last4=("4321",), district_norm="lckn",
        )
        c4_close = IdentityClues(
            unit_id="d::doc", file_rel="d", loc="doc",
            patient_names=(NameClue("Meena Rawat", ("meena", "rawat"), ("mn", "rwt")),),
            aadhaar_last4=("4321",), district_norm="lckn", age=42,
        )
        cmap_far = link_records([c1, c2_far, c3_missing_age])
        self.assertTrue(cmap_far["a::doc"] is None or cmap_far["a::doc"] != cmap_far["b::doc"])
        self.assertTrue(cmap_far["a::doc"] is None or cmap_far["a::doc"] != cmap_far["c::doc"])

        cmap_close = link_records([c1, c4_close])
        self.assertIsNotNone(cmap_close["a::doc"])
        self.assertEqual(cmap_close["a::doc"], cmap_close["d::doc"])


if __name__ == "__main__":
    unittest.main()
