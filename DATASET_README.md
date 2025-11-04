# SDBench Dataset Construction Guide

This document explains how we convert the original Parquet into a SDBench-ready dataset and the exact field mappings to `CaseFile`.

## Source → Converted → Adapted

1. Source Parquet (example):
   - `/Users/yufei/Desktop/SDBench/shzyk/DiagnosisArena/data/test-00000-of-00001.parquet`
   - Schema (abridged):
     - `id: int64`
     - `Case Information: string`
     - `Physical Examination: string`
     - `Diagnostic Tests: string`
     - `Final Diagnosis: string`
     - `Options: struct<A: string, B: string, C: string, D: string>`
     - `Right Option: string`

2. Converted (flatten Options A–D):
   - Script: `scripts/convert_parquet.py`
   - Output (in `/Users/yufei/Desktop/SDBench/converted`):
     - `test-00000-of-00001.jsonl`
     - `test-00000-of-00001.csv`
     - `test-00000-of-00001.sample100.jsonl`

3. Adapted to SDBench `CaseFile`:
   - Loader: `data_loader.py`
   - Builder CLI: `scripts/build_dataset.py`
   - Output:
     - `test-00000-of-00001.sdbench.jsonl`
     - `test-00000-of-00001.sdbench.csv`

## Field Mapping (Converted JSONL → CaseFile)

- `case_id`: `DA_{id}` (if `id` missing, use line number)
- `initial_abstract`: first 2 sentences (≤ ~360 chars) from `case_information` → fallback `physical_examination` → fallback `diagnostic_tests`
- `full_case_text` (stitched sections):
  - `PRESENTATION OF CASE` + `case_information`
  - `PHYSICAL EXAMINATION` + `physical_examination`
  - `DIAGNOSTIC TESTS` + `diagnostic_tests`
  - `OPTIONS` with lines `A. ... / B. ... / C. ... / D. ...` (only if present)
- `ground_truth_diagnosis`: `final_diagnosis`
- `publication_year`: provided via CLI (default 2025)
- `is_test_case`: provided via CLI (`--is_test_case` flag)

Notes:
- `Right Option` is not used by SDBench (we evaluate free-text diagnosis). It is preserved implicitly in the original converted files if needed for external tasks.

## Commands

### A) Convert Parquet → Flattened JSONL/CSV
```bash
python /Users/yufei/Desktop/SDBench/scripts/convert_parquet.py \
  "/Users/yufei/Desktop/SDBench/shzyk/DiagnosisArena/data/test-00000-of-00001.parquet" \
  --outdir "/Users/yufei/Desktop/SDBench/converted"
```
Generated:
- `converted/test-00000-of-00001.jsonl`
- `converted/test-00000-of-00001.csv`
- `converted/test-00000-of-00001.sample100.jsonl`

### B) Build SDBench-ready Dataset (CaseFile JSONL/CSV)
```bash
python /Users/yufei/Desktop/SDBench/scripts/build_dataset.py \
  "/Users/yufei/Desktop/SDBench/converted/test-00000-of-00001.jsonl" \
  --outdir "/Users/yufei/Desktop/SDBench/converted" \
  --publication_year 2025 \
  --is_test_case
```
Generated:
- `converted/test-00000-of-00001.sdbench.jsonl` (authoritative input for SDBench)
- `converted/test-00000-of-00001.sdbench.csv` (human-readable)

### Optional: Programmatic Loading
```python
from data_loader import load_jsonl_cases
cases = load_jsonl_cases(
    "/Users/yufei/Desktop/SDBench/converted/test-00000-of-00001.jsonl",
    publication_year=2025,
    is_test_case=True,
)
```

## Quality Checks
- Verify JSONL line count equals expected number of rows in source.
- Spot-check a few rows: `initial_abstract` is short and coherent.
- Ensure `full_case_text` contains PRESENTATION/EXAM/TESTS/OPTIONS sections as available.
- Confirm `ground_truth_diagnosis` matches original `Final Diagnosis`.

## Integration with SDBench
Use `*.sdbench.jsonl` to instantiate cases and run benchmarks:
```python
from data_loader import load_jsonl_cases
from sdbench import SDBench
from config import Config
from example_agents import RandomDiagnosticAgent

cases = load_jsonl_cases(
    "/Users/yufei/Desktop/SDBench/converted/test-00000-of-00001.jsonl",
    publication_year=2025,
    is_test_case=True,
)
bench = SDBench(Config())
result = bench.run_benchmark(RandomDiagnosticAgent("Random"), cases[:5], max_turns_per_case=10)
```

## Dependencies
- Conversion requires `pyarrow` (preferred) or `pandas`.
- CSV export in `data_loader.py` requires `pandas`.

If needed:
```bash
pip install pyarrow pandas
```
