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

For standalone Person 2 pipeline benchmarking prior to final Sync 2 merge:
```bash
RAKSHA_SECRET=practice python dev_p2_main.py --input ../dev/input --output ../out_p2
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
  ├─► [2] Cross-Script Clue Extraction (translit.py + clue_extractor.py)
  │     • Hindi-aware Devanagari→Latin transliteration & phonetic skeletons
  │     • Token-boundary key classification & per-column DMY/MDY/ISO inference
  │     • Bilingual free-text, JSON, CSV, and FHIR R4 Bundle extraction
  │
  ├─► [3] Precision-First Record Linkage (linkage.py)
  │     • Deterministic inverted-index blocking
  │     • Tiered hard-ID + composite soft links (DOB, relative, village+age)
  │     • Homonym, generic-token, and anti-transitive contradiction guards
  │
  ├─► [4] Adaptive Privacy Layer — Option C (privacy.py + policy.yaml)
  │     • Keyed HMAC-SHA256 pseudonyms (PSN-XXXXXX) & per-patient date shifts
  │     • Truthful 5-yr age bands, District + 3-digit PIN prefix on normal cells
  │     • Adaptive widening (10/20-yr) & QI suppression on k<5 / rare conditions
  │
  ├─► [5] Multilingual PII Scrubbing & Format Export (detector.py + formats.py)
  │     • Context-aware PII detection & meaning-trap guards (Dr., doses, C/O)
  │     • Filename, header, JSON key, metadata & base64 payload scrubbing
  │
  └─► [6] Release Assembly (security.py)
        • sanitized/ (UTF-8) + records.jsonl + honest report.json
```

---

## 3. Core Components & Algorithms

### Cross-Script Transliteration (`rakshak/translit.py`)
Combines `indic-transliteration` (ITRANS) with Hindi-specific schwa deletion, anusvara assimilation, and nukta/retroflex rules (`.D` → `d`, `.Dh` → `rh`, `K` → `kh`, `J` → `z`), followed by consonant-skeleton and `jellyfish` phonetic keys. Generic tokens (`devi`, `kumar`, `kumari`, `singh`) never link independently, and honorifics (`ji`, `sahab`, `babu`, `amma`, `didi`, `श्री`) are stripped before comparison.

### Clue Extraction & Date Inference (`rakshak/clue_extractor.py`)
Infers `DMY` vs. `MDY` vs. `ISO` date ordering per `(file, column)` before parsing (handling `DD/MM/YYYY`, `MM/DD/YYYY`, `DD.MM.YYYY`, `YYYY-MM-DD`, and Devanagari digits `०-९`). Extracts structured, FHIR, and free-text clues while masking 14-digit ABHA IDs prior to 12-digit Aadhaar/phone scans and ignoring `Dr.` lines, clinical durations (`x 8 yrs`), and `C/O <complaint>` phrases.

### Precision-First Record Linkage (`rakshak/linkage.py`)
Avoids false merges via two-tier evidence scoring and an anti-transitive Union-Find guard: merges are blocked if candidate clusters conflict on `gender`, `dob`, `aadhaar`, `abha`, `age` spread (`> 3` years), or multi-token full names. Re-registrations with multiple UHIDs at the same hospital bridge via phone/DOB/relative links, while `DOB + record_date` arithmetic resolves multi-year age drift cleanly.

### Adaptive k-Anonymity Privacy Layer (`rakshak/privacy.py`)
Implements **Option C (Adaptive k-Anonymity Widening)** driven by `config/policy.yaml`:
- **Pseudonyms & Dates:** Deterministic `HMAC-SHA256(RAKSHA_SECRET, ...)` issues `PSN-XXXXXX` (written both inside records and in `records.jsonl`) and non-zero per-patient shifts in `[-180, +180]` days, preserving exact length-of-stay and inter-visit intervals while defeating global-shift recovery.
- **Adaptive Generalization:** Normal equivalence classes `(age_band, gender, district)` (`k >= 5`, common diagnoses) keep 5-year age bands, District, and `3`-digit PIN prefixes (`226***`). Small cells (`k < 5`) or rare-diagnosis (`<= 3` patients) / elderly (`>= 85`) profiles widen to 10/20-year age bands (`80+`) and suppress PIN prefix / District *without* altering diagnoses, doses, labs, or stay lengths.

---

## 4. Security Design

- **Encryption at Rest:** `EphemeralVault` (`rakshak/security.py`) encrypts quarantined raw files as `quarantine/<sha256>.enc` using `AES-256-GCM` (`cryptography`) with an ephemeral 256-bit in-memory key and `AAD = input_sha256`.
- **Key Hygiene:** `RAKSHA_SECRET` is used exclusively in memory for keyed HMACs and never written to outputs, reports, or logs (`assert_no_secret_in_bytes`).
- **PII-Free Logging & Determinism:** `stdout` emits integer counters only; file traversal, blocking buckets, edge scoring, and JSON keys are sorted deterministically.

---

## 5. Declared Libraries & Model Policy

- **No LLMs / No Generative Models / No >50M-Parameter Transformers.**
- **Allowed Offline CPU Libraries (`requirements.txt`):** `rapidfuzz`, `jellyfish`, `indic-transliteration`, `cryptography`, `python-dateutil`, `PyYAML` (plus Python 3.10+ standard library).

---

## 6. Measured Results (`dev/input`, Person 2 Pipeline + Local Attacker Audit)

| Metric | Value |
|---|---|
| Patient Linking (Precision / Recall / F0.5 / Wrong Merges) | `1.00 / 1.00 / 1.00 / 0` |
| Medical Values Kept (`check_dev.py`) | `4847 / 4855 (99.8%)` (0 clinical probes lost) |
| Age-Band Truthfulness / Stay-Length Exactness | `359/359 (100%)` / `65/65 (100%)` |
| QI-Only Re-identification Risk (A4 ≥ 0.5, `test_attack_risk.py`) | `0.0208` (vs `0.9792` raw) |
| Global Shift Recovered (A5) / Pseudonym Inversion (A6) | `False` / `0` |
| Combined Full PII Leak & Chaos Benchmark | *[Pending Sync 2 integration with Person 1]* |

---

## 7. Known Limitations

- **Single-token uncorroborated notes:** Notes containing only a generic name or single name without relative/village/ID clues intentionally abstain (`patient: null`) to prevent homonym false merges.
- **Unique rare diagnoses:** Keeping rare ICD-10 conditions intact preserves clinical utility (`clinical_probes`), meaning a known-fact attacker targeting a globally unique diagnosis + exact stay length retains residual targeted signal even after district/age suppression.
- **Unstructured OCR/binary files:** Unsupported binaries (`.pdf`, `.jpg`, `.xml`) are quarantined encrypted at rest rather than OCR-processed offline.
