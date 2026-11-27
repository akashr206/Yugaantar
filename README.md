# Team `enma` · Rakshak Privacy Firewall

**Yugaantar DevForge · Track 3: Secret Wards**  
Offline, CPU-only, LLM-free multilingual privacy firewall for heterogeneous Indian hospital records (English, Devanagari Hindi, and Hinglish).

---

## 1. Setup & Usage

### Install Dependencies (One-time Setup)
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run Offline
```bash
RAKSHA_SECRET=practice python main.py --input ../dev/input --output ../out_dev
```

---

## 2. System Architecture

```text
Input Tree (CSV / TSV / JSON / FHIR R4 / TXT / Misc)
  │
  ├─► [1] Ingestion & Quarantine (ingest.py + security.EphemeralVault)
  │     • Sniffs format & decodes UTF-8 / UTF-8-BOM / UTF-16 / cp1252
  │     • Encrypts unsupported/corrupt files (PDF/JPG/XML/empty/broken JSON)
  │       at rest into quarantine/<sha256>.enc via AES-256-GCM
  │
  ├─► [2] Cross-Script Clue Extraction (translit.py + schema_profiler.py + clue_extractor.py)
  │     • Hindi-aware Devanagari→Latin transliteration, phonetic & symmetric abbreviation matching
  │     • Unified token-boundary key classification & per-column DMY/MDY/ISO inference
  │     • Facility-agnostic bilingual free-text, JSON, CSV, and FHIR R4 Bundle extraction
  │
  ├─► [3] Precision-First Record Linkage (linkage.py)
  │     • Deterministic inverted-index blocking
  │     • Tiered hard-ID + composite soft links (DOB, relative, village+age)
  │     • Homonym, generic-token, and anti-transitive contradiction guards
  │
  ├─► [4] Balanced Hybrid Adaptive Privacy Layer (`planner.py` → `privacy.py` + `policy.yaml`)
  │     • Keyed HMAC-SHA256 pseudonyms (PSN-XXXXXX) & per-patient date shifts
  │     • Truthful sliding <=30-yr / N+ age bands maximizing diagnosis & stay peer overlap
  │     • Surgical Gender suppression only on small mixed-sex cells; DOB masking on widened/suppressed clusters
  │
  ├─► [5] Multilingual PII Scrubbing & Guarded CRF (`detector.py` + `ner_crf.py` + `formats.py`)
  │     • Context-aware PII detection & meaning-trap guards (Dr., doses, eponyms, C/O)
  │     • Compact offline python-crfsuite Stage 3.5 fallback NER (`models/pii_ner.crfsuite`)
  │     • Filename, header, JSON key, metadata, FHIR display & base64 payload scrubbing
  │
  └─► [6] Release Assembly (security.py)
        • sanitized/ (UTF-8) + records.jsonl + honest report.json
```

---

## 3. Core Components & Algorithms

### Cross-Script Transliteration & Generalized Name Matching (`rakshak/translit.py`)
Combines `indic-transliteration` (ITRANS) with Hindi-specific schwa deletion, anusvara assimilation, and nukta/retroflex rules (`.D` → `d`, `.Dh` → `rh`, `K` → `kh`, `J` → `z`), followed by consonant-skeleton and `jellyfish` phonetic keys. Unseen Indian/hospital honorifics (`shri`, `smt`, `kumari`, `ji`, `sahab`, `babu`, `amma`, `didi`, `bhai`, etc.), dotted initials (`R. K. Verma` ↔ `Ramesh Kumar Verma`), and consonant contractions (`Sk` ↔ `Shaikh`, `Kr` ↔ `Kumar`, `Pd` ↔ `Prasad`, `Bhatt` ↔ `Bhattacharya`) are matched symmetrically (`len >= 2`), while generic single tokens never link independently.

### Unified Schema Profiling & Clue Extraction (`rakshak/schema_profiler.py`, `rakshak/clue_extractor.py`)
Classifies columns and JSON keys cleanly across unseen datasets (`beneficiary`, `member id`, `subscriber`, `kin`, `nominee`, `tehsil`/`mandal`/`taluka`, `umra`, `रोगी`) while guarding clinical chemistry/domain phrases (`Serum Creatinine`, `Village/Mohalla`). Infers `DMY` vs. `MDY` vs. `ISO` date ordering per `(file, column)` (handling `DD/MM/YYYY`, `MM/DD/YYYY`, `Month DD, YYYY`, Hindi months, and Devanagari digits `०-९`), shields pediatric `/` values `< 20` against false slash-date linkage, masks 14-digit ABHA IDs before Aadhaar/phone scans, and excludes `clean/` protocol/formulary decoys structurally.

### Precision-First Record Linkage (`rakshak/linkage.py`)
Avoids false merges via two-tier evidence scoring and an anti-transitive Union-Find guard: merges are blocked if candidate clusters conflict on `gender`, `dob`, `aadhaar`, `abha`, `age` spread (`> 3` years), or multi-token full names. Tier-1 strong ID matches allow verified initial/contraction variants while multi-UHID visits at the same facility bridge via phone/DOB/relative links.

### Balanced Hybrid Privacy Layer (`rakshak/privacy.py`)
Implements **Balanced Hybrid Adaptive Generalization** driven by `config/policy.yaml`:
- **Pseudonyms & Dates:** Deterministic `HMAC-SHA256(RAKSHA_SECRET, ...)` issues `PSN-XXXXXX` (written both inside records and in `records.jsonl`) and non-zero per-patient shifts in `[-180, +180]` days, preserving exact length-of-stay and inter-visit intervals while defeating global-shift recovery.
- **Truthful Sliding Age Bands & Surgical Gender Suppression:** Standard equivalence classes (`k >= 5`, common diagnoses) keep `5`-year age bands, `Gender`, `District`, and `3`-digit PIN prefixes. Widened/suppressed or rare-condition clusters pick the truthful `<= 30`-year sliding window (`0-29`..`70-99`) or open-ended `N+` band (`50+`/`60+`/`70+`/`80+`) that maximizes shared diagnosis/stay peer overlap (never emitting uninformative `[AGE]`), mask `DOB` (`[DOB]` / omit `Patient.birthDate` in FHIR) so Sweeney filters cannot recover birth years, and suppress `Gender` surgically **only** when the mixed-sex pool strictly enlarges a `<= 2`-person same-sex cohort.

### Guarded Offline CRF Fallback (`rakshak/detector.py`, `rakshak/ner_crf.py`)
Rule/anchor patterns handle arbitrary unseen facility names, structural MRNs, clinical medication doses (`<Drug> <dose> OD/BD`), and bilingual addresses (`जिला:`/`जनपद`). When rule anchors miss unanchored free-text person names, Stage 3.5 invokes a compact `165.8 KB` (`2,822` learned weights) anti-overfit `python-crfsuite` linear-chain CRF (`models/pii_ner.crfsuite`) without open-vocabulary name memorization, gated by `min_prob = 0.78`, `Dr.`/disease-eponym shields, and clinical/schema stopword filters, followed by Stage 3.6 linked-cluster surface redaction (`build_cluster_redact_context`).

---

## 4. Security Design

- **Encryption at Rest:** `EphemeralVault` (`rakshak/security.py`) encrypts quarantined raw files as `quarantine/<sha256>.enc` using `AES-256-GCM` (`cryptography`) with an ephemeral 256-bit in-memory key and `AAD = input_sha256`.
- **Key Hygiene:** `RAKSHA_SECRET` is used exclusively in memory for keyed HMACs and never written to outputs, reports, or logs (`assert_no_secret_in_bytes`).
- **PII-Free Logging & Determinism:** `stdout` emits integer counters only; file traversal, blocking buckets, edge scoring, and JSON keys are sorted deterministically.

---

## 5. Declared Libraries & Model Policy

- **No LLMs / No Generative Models / No >50M-Parameter Transformers.**
- **Allowed Offline CPU Libraries (`requirements.txt`):** `rapidfuzz`, `jellyfish`, `indic-transliteration`, `cryptography`, `python-dateutil`, `PyYAML`, `python-crfsuite>=0.9.10` (`models/pii_ner.crfsuite`, `165.8 KB` / `2,822`-parameter linear-chain CRF trained purely offline via `scripts/train_ner_crf.py` on external/synthetic corpora, excluding all dev/eval data, plus Python 3.10+ standard library).

---

## 6. Measured Results (`dev/input`, Local Attack Audit, External & OOD Benchmarks)

| Metric | Value |
|---|---|
| Personal Data Leaked (`check_dev.py`) | `0 / 3230 (0.0%)` (Latin `0.0%`, Deva `0.0%`, Hinglish `0.0%`; all hidden channels `0.0%`) |
| Planted Canaries / Raw Broken Copies | `0 / 4` canaries / `0` raw copies (`8 / 8` corrupt/unsupported files encrypted at rest) |
| Patient Linking (Precision / Recall / F0.5 / Wrong Merges) | `1.00 / 1.00 / 1.00 / 0` (`777 / 821` records linked across `192` sanitized files) |
| Medical Values Kept (`check_dev.py`) | `4762 / 4855 (98.1%)` (`100.0%` of clinical probes kept; suppressed rare/small-cell Districts omitted) |
| Age-Band Truthfulness / Stay-Length Exactness | `359/359 (100%)` / `65/65 (100%)` |
| Profile Re-identification Risk (A4 ≥ 0.5, `test_attack_risk.py`) | **`0.0000` (`0 / 48`)** (`full == QI-only`; mean Sweeney prob **`0.0122`**; A5 targeted: **`0.3571`**) |
| Global Shift Recovered (A5) / A6 Inversion / A7 Cross-Release | `False` / `0` / `0` shared pseudonyms (`0/51`) & `0` positional date collisions (`0/1108`) |
| Runtime / Peak RSS (`report.json`, 200 files) | `2.53 s` / `56.1 MB` (`check_submission.py --run`: `3.0 s`, `READY TO SUBMIT`) |
| Unit, Chaos & Integration Suite (`unittest`) | **`48 / 48` passed (`11.235 s`)** |
| External Multi-Corpus Audit (`combined_benchmark`, 229 files) | Maskara `PERSON_NAME` `61/83 (73.5%)`, `UPI_ID` `18/18 (100%)`, `AADHAAR`/`PAN`/`EMAIL` `100%`; `clinical_traps` `Dr.` `10/10` & eponyms `10/10` |
| Independent Synthetic OOD Stress Corpus (`13` files, zero dev/train overlap) | `0 / 89 (0.0%)` PII leaked, `9/9` clinical terms kept, `8/8` cross-file abbreviated links, `2/2` homonyms separated, standalone CRF OOD ablation `9/10` vs `0/10` rules-only (`5/5` clinical negatives intact) |

*Note: `check_dev.py` maps renamed files via `input_sha256` against Linux LF hashes in `truth.json`; on Windows checkouts with Git CRLF conversion, restoring LF line endings or SHA-aligning text paths gives the exact 200-file alignment reported above.*

---

## 7. Known Limitations

- **Single-token uncorroborated notes:** Notes containing only a generic name or single name without relative/village/ID clues intentionally abstain (`patient: null`) to prevent homonym false merges.
- **Unique rare diagnoses vs. A5 residual (`0.3571`):** Balanced Hybrid cuts A5 targeted known-facts success from `0.5714` to `0.3571` while preserving every clinical diagnosis; residual A5 hits occur only on globally unique conditions combined with exact hospital stay lengths, which cannot be merged without distorting clinical utility.
- **Unanchored non-clinical synthetic tokens:** In external Maskara web/API logs (`API_KEY`, `IP_ADDRESS`, `USERNAME`, synthetic `00xx` landlines), non-healthcare tokens without clinical anchors remain outside hospital PII scope unless structurally matched.
- **Unstructured OCR/binary files:** Unsupported binaries (`.pdf`, `.jpg`, `.xml`) are quarantined encrypted at rest rather than OCR-processed offline.
