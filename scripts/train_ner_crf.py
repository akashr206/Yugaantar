#!/usr/bin/env python3
"""Offline trainer for the lightweight python-crfsuite fallback NER model.

Trains strictly on non-dev public Indian PII corpora (Maskara `test.jsonl` +
Hindi-Awadhi `sample_500.jsonl` lines 100..499) combined with disjoint synthetic
English, Hinglish, and Devanagari clinical prose, case augmentations, dotted
initials, multi-label PER/REL/LOC/ID spans, and hard clinical/org negatives.
Zero `dev/` or `ood_overfit_eval/` records are touched.
"""
from __future__ import annotations

import difflib
import json
import random
import re
import sys
from pathlib import Path

import pycrfsuite

YUG_ROOT = Path(__file__).resolve().parent.parent
KIT_ROOT = YUG_ROOT.parent
sys.path.insert(0, str(YUG_ROOT))

from rakshak.ner_crf import extract_sentence_features, tokenize_with_spans  # noqa: E402

MASKARA_TRAIN = KIT_ROOT / "external_datasets" / "maskara_indian_pii_200k" / "test.jsonl"
AWADHI_TRAIN = KIT_ROOT / "external_datasets" / "english_hindi_awadhi_deidentification" / "sample_500.jsonl"
MODEL_OUT = YUG_ROOT / "models" / "pii_ner.crfsuite"

# Disjoint synthetic names (including -vir/-gram endings, dotted initials, hyphen/apostrophe, 3-part names)
_FIRST_EN = [
    "Karan", "Priya", "Vikram", "Ananya", "Sandeep", "Deepak", "Pooja", "Manish",
    "Nandini", "Harish", "Meera", "Gaurav", "Shreya", "Abhishek", "Swati", "Nitin",
    "Kiran", "Sneha", "Pradeep", "Bhavna", "Tarun", "Divya", "Alok", "Ritika",
    "Rohan", "Kavita", "Rajesh", "Sunita", "Mukesh", "Archana", "Saurabh", "Preeti",
    "Ranvir", "Rajvir", "Sangram", "Karamvir", "Devendra", "Anne-Marie", "Qamar", "Farhan",
]
_LAST_EN = [
    "Malhotra", "Nair", "Bhattacharya", "Deshmukh", "Kulkarni", "Chatterjee",
    "Bansal", "Chopra", "Saxena", "Joshi", "Rao", "Iyer", "Menon", "Agarwal",
    "Sen", "Banerjee", "Patwardhan", "Gokhale", "Trivedi", "Kapur", "Verma", "Shukla",
    "D'Souza", "Gill", "Nambiar", "Barman", "Chhabra", "Sengupta",
]
_FIRST_HI = [
    "करण", "प्रिया", "विक्रम", "अनन्या", "संदीप", "दीपक", "पूजा", "मनीष",
    "नंदिनी", "हरीश", "मीरा", "गौरव", "श्रेया", "अभिषेक", "स्वाति", "नितिन",
    "किरण", "स्नेहा", "प्रदीप", "भावना", "तरुण", "दिव्या", "आलोक", "रोहन",
    "रणवीर", "संग्राम", "देवेंद्र", "कमर",
]
_LAST_HI = [
    "मल्होत्रा", "नायर", "भट्टाचार्य", "देशमुख", "कुलकर्णी", "चटर्जी",
    "बंसल", "चोपड़ा", "सक्सेना", "जोशी", "राव", "अग्रवाल", "बनर्जी", "शुक्ला", "त्रिवेदी",
    "बर्मन", "तिवारी", "गिल",
]
_LOC_PHRASES_EN = [
    "Mohalla Shivpuri", "Gali No 4 Azad Nagar", "Near Old Bus Stand Rampur",
    "Plot 18 Shastri Vihar", "Lane 3 Civil Lines", "Village Sitapur Kalan",
]
_LOC_PHRASES_HI = [
    "मोहल्ला शिवपुरी", "ग्राम सीतापुर कलां", "नजदीक पुराना बस अड्डा", "गली नंबर 4 आज़ाद नगर",
]
_ID_SAMPLES = [
    "KGMU/26/00912", "CMC-2026-4412", "ZLH_26_7731", "SGPGI-882210", "IP-26-00419", "NIMS/25/11890",
]
_DOCS_EN = ["Dr. S. Kulkarni", "Dr. Meenakshi Iyer", "Dr. P. K. Deshmukh", "Dr. A. Sengupta", "Dr. R. Subramanian"]
_DOCS_HI = ["डॉ. एस. कुलकर्णी", "डॉ. मीनाक्षी अय्यर", "डॉ. पी. के. देशमुख", "डॉ. ए. सेनगुप्ता"]
_HOSPITALS_EN = [
    "Apollo Specialty Hospital", "St. Jude Charitable Clinic", "District Hospital Varanasi",
    "Sunrise Diagnostics Lab", "Apex Heart Institute", "Kaveri Medical Centre",
]
_ORG_TRAPS = [
    "Automated Billing Switch", "Southern Rail Corridor", "Central Audit Directorate",
    "National Biometric Gateway", "Comprehensive Referral Letter", "State Health Regulatory Board",
]
_EPONYMS = [
    "Parkinson's disease", "Addison's disease", "Bell's palsy", "Crohn's disease",
    "Hodgkin lymphoma", "Alzheimer's disease", "Graves' disease", "Marfan syndrome",
]
_MEDS_LABS = [
    "Tab. Augmentin 625 mg BD", "Inj. Ceftriaxone 1 gm IV", "Cap. Omeprazole 20 mg OD",
    "Metformin 500 mg BD", "Serum Creatinine 1.1 mg/dL", "Blood Pressure 120/80 mmHg",
    "Random Blood Sugar 138 mg/dL", "Hemoglobin 12.4 g/dL", "Chief Complaint Chest Pain",
    "Complete Blood Count Normal", "Follow up after 5 days", "ASHA worker visited family",
]

_MASKARA_LABEL_MAP = {
    "PERSON_NAME": "PER",
    "PER": "PER",
    "PATIENT_NAME": "PER",
    "ADDRESS": "LOC",
    "AADHAAR": "ID",
    "PAN_CARD": "ID",
    "DRIVER_LICENSE": "ID",
    "PASSPORT": "ID",
    "SSN": "ID",
}
_AWADHI_TAG_MAP = {
    "[FULL_NAME]": "PER",
    "[NAME]": "PER",
    "[ADDRESS]": "LOC",
    "[CITY]": "LOC",
    "[MRN]": "ID",
    "[NATIONAL_ID]": "ID",
    "[PASSPORT_NUMBER]": "ID",
}


def _spans_to_bio(line: str, ent_spans: list[tuple[Any, ...]]) -> tuple[list[str], list[str]] | None:
    tok_spans = tokenize_with_spans(line)
    if len(tok_spans) < 2:
        return None
    toks: list[str] = []
    labels: list[str] = []
    prev_ent_idx = -1
    prev_ent_type = ""
    for s, e, tok in tok_spans:
        toks.append(tok)
        matched_idx = -1
        matched_type = ""
        for idx, item in enumerate(ent_spans):
            es, ee = item[0], item[1]
            etype = item[2] if len(item) > 2 else "PER"
            if s < ee and e > es:
                matched_idx = idx
                matched_type = etype
                break
        if matched_idx >= 0 and re.search(r"[^\W_.]|^\.$", tok):
            # Keep salutations/role prefixes as O even if gold span touched them
            if matched_type in ("PER", "REL") and tok.lower().rstrip(".") in (
                "mr", "mrs", "ms", "miss", "dr", "prof", "shri", "smt", "श्री", "श्रीमती",
                "policyholder", "passenger", "claimant", "subscriber", "bhai",
            ):
                labels.append("O")
                prev_ent_idx = -1
                prev_ent_type = ""
            elif tok in (".", ",") and matched_idx != prev_ent_idx:
                labels.append("O")
                prev_ent_idx = -1
                prev_ent_type = ""
            elif matched_idx == prev_ent_idx and matched_type == prev_ent_type:
                labels.append(f"I-{matched_type}")
            else:
                labels.append(f"B-{matched_type}")
                prev_ent_idx = matched_idx
                prev_ent_type = matched_type
        else:
            labels.append("O")
            prev_ent_idx = -1
            prev_ent_type = ""
    return toks, labels


def load_maskara_sentences(
    rng: random.Random,
    limit: int = 2000,
    return_sources: bool = False,
) -> list[Any]:
    if not MASKARA_TRAIN.is_file():
        return []
    pos_sents: list[tuple[list[str], list[str], str]] = []
    neg_sents: list[tuple[list[str], list[str], str]] = []
    seen_lines: set[str] = set()
    for raw_line in MASKARA_TRAIN.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        doc = json.loads(raw_line)
        text = doc.get("text", "")
        ents: list[tuple[int, int, str]] = []
        for e in doc.get("entities") or []:
            mapped = _MASKARA_LABEL_MAP.get(str(e.get("label") or ""))
            if mapped:
                ents.append((int(e["start"]), int(e["end"]), mapped))
        off = 0
        for ln in text.splitlines(keepends=True):
            ln_s = ln.rstrip("\r\n")
            key = " ".join(ln_s.casefold().split())
            if 5 <= len(ln_s.strip()) <= 260 and key not in seen_lines:
                seen_lines.add(key)
                local_ents = [(es - off, ee - off, et) for es, ee, et in ents if es >= off and ee <= off + len(ln)]
                bio = _spans_to_bio(ln_s, local_ents)
                if bio:
                    src_tag = f"maskara_{doc.get('domain', 'gen')}"
                    if local_ents:
                        pos_sents.append((bio[0], bio[1], src_tag))
                    else:
                        neg_sents.append((bio[0], bio[1], src_tag))
            off += len(ln)
    rng.shuffle(pos_sents)
    rng.shuffle(neg_sents)
    combined = pos_sents[:limit] + neg_sents[: limit // 3]
    if return_sources:
        return combined
    return [(t, l) for t, l, _ in combined]


def load_awadhi_hindi_sentences(return_sources: bool = False) -> list[Any]:
    """Use held-out tail lines 100..499 (distinct from combined_benchmark aw_000..099)."""
    if not AWADHI_TRAIN.is_file():
        return []
    lines = [l for l in AWADHI_TRAIN.read_text(encoding="utf-8").splitlines() if l.strip()]
    out: list[tuple[list[str], list[str], str]] = []
    for raw in lines[100:]:
        rec = json.loads(raw)
        inp = str(rec.get("input", ""))
        out_txt = str(rec.get("output", ""))
        if not any(tag in out_txt for tag in _AWADHI_TAG_MAP):
            continue
        in_toks = tokenize_with_spans(inp)
        out_toks = tokenize_with_spans(out_txt)
        in_words = [t for _, _, t in in_toks]
        out_words = [t for _, _, t in out_toks]
        sm = difflib.SequenceMatcher(None, in_words, out_words)
        labels = ["O"] * len(in_words)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "replace":
                mapped_type = None
                for ow in out_words[j1:j2]:
                    if ow in _AWADHI_TAG_MAP:
                        mapped_type = _AWADHI_TAG_MAP[ow]
                        break
                if mapped_type:
                    first = True
                    for k in range(i1, i2):
                        if re.search(r"[^\W_]", in_words[k]):
                            labels[k] = f"B-{mapped_type}" if first else f"I-{mapped_type}"
                            first = False
        if any(lbl != "O" for lbl in labels) and len(in_words) >= 2:
            out.append((in_words, labels, "awadhi_tail"))
    if return_sources:
        return out
    return [(t, l) for t, l, _ in out]


def build_synthetic_clinical_data(
    rng: random.Random,
    n_samples: int = 3600,
    return_sources: bool = False,
) -> list[Any]:
    data: list[tuple[list[str], list[str], str]] = []

    def _add_tagged(parts: list[tuple[str, str | None]], tmpl_tag: str, case_aug: bool = True) -> None:
        raw_parts = list(parts)
        mode = rng.random() if case_aug else 0.0
        # 16% lowercase chat augmentation, 10% ALL-CAPS SMS augmentation on English/Hinglish text
        has_deva = any(re.search(r"[\u0900-\u097F]", c) for c, _ in raw_parts)
        if not has_deva:
            if mode < 0.16:
                raw_parts = [(c.lower(), t) for c, t in raw_parts]
            elif mode < 0.26:
                raw_parts = [(c.upper(), t) for c, t in raw_parts]

        line = ""
        spans: list[tuple[int, int, str]] = []
        for chunk, etype in raw_parts:
            s = len(line)
            line += chunk
            e = len(line)
            if etype:
                spans.append((s, e, etype))
        bio = _spans_to_bio(line, spans)
        if bio:
            data.append((bio[0], bio[1], tmpl_tag))

    for idx in range(n_samples):
        f_en = rng.choice(_FIRST_EN)
        m_en = rng.choice(_FIRST_EN)
        l_en = rng.choice(_LAST_EN)
        full_en = f"{f_en} {l_en}"
        tri_en = f"{f_en} {m_en} {l_en}"
        init_en = f"{f_en[0]}. {m_en[0]}. {l_en}"
        f2_en = f"{rng.choice(_FIRST_EN)} {rng.choice(_LAST_EN)}"
        f_hi = rng.choice(_FIRST_HI)
        l_hi = rng.choice(_LAST_HI)
        full_hi = f"{f_hi} {l_hi}"
        tri_hi = f"{f_hi} {rng.choice(_FIRST_HI)} {l_hi}"
        doc_en = rng.choice(_DOCS_EN)
        doc_hi = rng.choice(_DOCS_HI)
        hosp = rng.choice(_HOSPITALS_EN)
        org_trap = rng.choice(_ORG_TRAPS)
        epon = rng.choice(_EPONYMS)
        med = rng.choice(_MEDS_LABS)
        loc_en = rng.choice(_LOC_PHRASES_EN)
        loc_hi = rng.choice(_LOC_PHRASES_HI)
        pid = rng.choice(_ID_SAMPLES)
        yr = str(rng.randint(2020, 2027))

        kind = idx % 22
        tmpl = f"synth_t{kind}"
        if kind == 0:
            _add_tagged([(full_en, "PER"), (" presented with fever and cough for 3 days.", None)], tmpl)
        elif kind == 1:
            _add_tagged([("Admitted ", None), (full_en, "PER"), (" to Ward 3 under ", None), (doc_en, None), (".", None)], tmpl)
        elif kind == 2:
            _add_tagged([("Referred by ", None), (f2_en, "REL"), (" for ", None), (full_en, "PER"), (" at ", None), (hosp, None), (".", None)], tmpl)
        elif kind == 3:
            _add_tagged([(f_en, "PER"), (", wife of ", None), (f2_en, "REL"), (", reported giddiness.", None)], tmpl)
        elif kind == 4:
            _add_tagged([("On January 12, ", None), (yr, None), (", ", None), (full_en, "PER"), (" reported severe headache.", None)], tmpl)
        elif kind == 5:
            _add_tagged([(yr, None), (", ", None), (full_en, "PER"), (" underwent review at ", None), (hosp, None), (".", None)], tmpl)
        elif kind == 6:
            _add_tagged([(init_en, "PER"), (" visited clinic on follow-up; vitals stable.", None)], tmpl, case_aug=False)
        elif kind == 7:
            _add_tagged([("Please recheck temp for ", None), (tri_en, "PER"), (" before evening round.", None)], tmpl)
        elif kind == 8:
            _add_tagged([("Afebrile this morning, ", None), (f_en, "PER"), (" ambulating well in corridor.", None)], tmpl, case_aug=False)
        elif kind == 9:
            # Hinglish clinical prose (absent in previous trainer!)
            _add_tagged([(full_en, "PER"), (" ko 3 din se tez bukhar aur khansi hai.", None)], tmpl)
        elif kind == 10:
            _add_tagged([("Subah ward mein ", None), (full_en, "PER"), (" ki report normal aayi, ", None), (f2_en, "REL"), (" ko bata diya.", None)], tmpl)
        elif kind == 11:
            _add_tagged([(f_en, "PER"), (" ji ne dawa samay par li aur ", None), (doc_en, None), (" se checkup karaya.", None)], tmpl)
        elif kind == 12:
            _add_tagged([("Policyholder ", None), (full_en, "PER"), (" registered under ", None), (pid, "ID"), (" at ", None), (org_trap, None), (".", None)], tmpl, case_aug=False)
        elif kind == 13:
            _add_tagged([("Passenger ", None), (tri_en, "PER"), (" residing at ", None), (loc_en, "LOC"), (" verified ", None), (pid, "ID"), (".", None)], tmpl, case_aug=False)
        elif kind == 14:
            _add_tagged([(full_hi, "PER"), (" को कल शाम से तेज बुखार और खांसी है।", None)], tmpl)
        elif kind == 15:
            _add_tagged([(tri_hi, "PER"), (" निवासी ", None), (loc_hi, "LOC"), (" को पिछले 5 दिनों से कमजोरी है।", None)], tmpl)
        elif kind == 16:
            _add_tagged([("मरीज ", None), (full_hi, "PER"), (" की जांच ", None), (doc_hi, None), (" ने की।", None)], tmpl)
        elif kind == 17:
            _add_tagged([(f_hi, "PER"), (" और ", None), (rng.choice(_FIRST_HI) + " " + l_hi, "REL"), (" सुबह वार्ड में आए।", None)], tmpl)
        elif kind == 18:
            _add_tagged([("pls check ", None), (full_en.lower(), "PER"), (" rx and update file ", None), (pid, "ID")], tmpl, case_aug=False)
        elif kind == 19:
            # Hard TitleCase institutional / organizational & clinical field-header negatives (must stay all O)
            if idx % 3 == 0:
                _add_tagged([(f"Case Status - stable | Panel Name: Complete Blood Count | Visit Type - Routine.", None)], tmpl, case_aug=False)
            elif idx % 3 == 1:
                _add_tagged([(f"Claim Status - Approved |Vital Signs - Normal | Follow-up Note verified.", None)], tmpl, case_aug=False)
            else:
                _add_tagged([(f"The {org_trap} routed audit memo from {hosp} regarding {med}.", None)], tmpl, case_aug=False)
        elif kind == 20:
            # Hard clinical negatives: dates, hospitals, doctors, eponyms, labs
            _add_tagged([(f"On April 09, {yr}, {doc_en} at {hosp} diagnosed {epon}; advised {med}.", None)], tmpl, case_aug=False)
        else:
            if idx % 2 == 0:
                _add_tagged([("During rounds in Endocrine Unit, ", None), (full_en, "PER"), (" requested discharge.", None)], tmpl)
            else:
                _add_tagged([("\x00SHIELD_0\x00 evaluated ", None), (full_en, "PER"), (" for \x00SHIELD_1\x00; ", None), (med, None)], tmpl)

    if return_sources:
        return data
    return [(t, l) for t, l, _ in data]


def main() -> int:
    rng = random.Random(20261127)
    maskara = load_maskara_sentences(rng, limit=2000, return_sources=True)
    awadhi = load_awadhi_hindi_sentences(return_sources=True)
    synth = build_synthetic_clinical_data(rng, n_samples=3600, return_sources=True)
    all_rows = maskara + awadhi + synth

    # Disjoint source/template-stratified 90/10 split so validation templates don't leak into training
    buckets: dict[str, list[tuple[list[str], list[str]]]] = {}
    for toks, labels, src in all_rows:
        buckets.setdefault(src, []).append((toks, labels))

    train_sents: list[tuple[list[str], list[str]]] = []
    val_sents: list[tuple[list[str], list[str]]] = []
    src_keys = sorted(buckets.keys())
    rng.shuffle(src_keys)
    # Hold out dedicated template/domain groups plus 10% tail of large corpora cleanly
    val_sources = {"synth_t5", "synth_t10", "synth_t13", "synth_t19"}
    for sk in src_keys:
        items = buckets[sk]
        if sk in val_sources:
            val_sents.extend(items)
        elif sk.startswith("synth_"):
            train_sents.extend(items)
        else:
            cut = max(1, int(len(items) * 0.9))
            train_sents.extend(items[:cut])
            val_sents.extend(items[cut:])

    rng.shuffle(train_sents)
    trainer = pycrfsuite.Trainer(verbose=False)
    for toks, labels in train_sents:
        trainer.append(extract_sentence_features(toks), labels)
    trainer.set_params(
        {
            "c1": 0.12,
            "c2": 0.25,
            "max_iterations": 150,
            "feature.possible_transitions": True,
        }
    )
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    trainer.train(str(MODEL_OUT))

    # Measure held-out validation token precision/recall across PER/REL/LOC/ID
    tagger = pycrfsuite.Tagger()
    tagger.open(str(MODEL_OUT))
    tp = fp = fn = 0
    for toks, gold in val_sents:
        pred = tagger.tag(extract_sentence_features(toks))
        for g, p in zip(gold, pred):
            g_is = g != "O"
            p_is = p != "O"
            if g_is and p_is and g.split("-")[-1] == p.split("-")[-1]:
                tp += 1
            else:
                if p_is:
                    fp += 1
                if g_is:
                    fn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    size_kb = MODEL_OUT.stat().st_size / 1024
    print(
        f"Trained {MODEL_OUT.relative_to(YUG_ROOT)} ({size_kb:.1f} KB) | "
        f"sents: train={len(train_sents)} (maskara={len(maskara)}, awadhi={len(awadhi)}, synth={len(synth)}), "
        f"val={len(val_sents)} | Entity tok P={prec:.3f} R={rec:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
