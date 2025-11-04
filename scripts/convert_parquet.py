import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def flatten_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    # Map potential column names (with/without spaces)
    def get_first(keys: List[str]):
        for k in keys:
            if k in row and row[k] is not None:
                return row[k]
        return None

    out["id"] = get_first(["id"])  # int64
    out["case_information"] = get_first(["Case Information", "case_information"])  # str
    out["physical_examination"] = get_first(["Physical Examination", "physical_examination"])  # str
    out["diagnostic_tests"] = get_first(["Diagnostic Tests", "diagnostic_tests"])  # str
    out["final_diagnosis"] = get_first(["Final Diagnosis", "final_diagnosis"])  # str

    options = get_first(["Options", "options"]) or {}
    if isinstance(options, dict):
        out["option_a"] = options.get("A")
        out["option_b"] = options.get("B")
        out["option_c"] = options.get("C")
        out["option_d"] = options.get("D")
    else:
        # If serialized JSON string
        try:
            obj = json.loads(options)
            out["option_a"] = obj.get("A")
            out["option_b"] = obj.get("B")
            out["option_c"] = obj.get("C")
            out["option_d"] = obj.get("D")
        except Exception:
            out["option_a"] = None
            out["option_b"] = None
            out["option_c"] = None
            out["option_d"] = None

    out["right_option"] = get_first(["Right Option", "right_option"])  # e.g., 'A'
    return out


def main():
    parser = argparse.ArgumentParser(description="Convert Parquet to CSV/JSONL with flattened Options (A-D)")
    parser.add_argument("input", type=str, help="Path to input parquet file")
    parser.add_argument("--outdir", type=str, default="converted", help="Output directory")
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit (0 = all)")
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.outdir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try pyarrow first, fallback to pandas if needed
    table = None
    engine = None

    try:
        import pyarrow.parquet as pq
        table = pq.read_table(in_path)
        engine = "pyarrow"
        data = table.to_pylist()
    except Exception as e:
        try:
            import pandas as pd
            df = pd.read_parquet(in_path)
            engine = "pandas"
            data = df.to_dict(orient="records")
        except Exception as e2:
            raise RuntimeError(f"Failed to read parquet. pyarrow_error={repr(e)}, pandas_error={repr(e2)}")

    if args.limit and args.limit > 0:
        data = data[: args.limit]

    flat_rows: List[Dict[str, Any]] = [flatten_row(r) for r in data]

    # Write JSONL
    jsonl_path = out_dir / f"{in_path.stem}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in flat_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write CSV
    try:
        import pandas as pd
        pd.DataFrame(flat_rows).to_csv(out_dir / f"{in_path.stem}.csv", index=False)
    except Exception:
        # Minimal CSV writer if pandas is unavailable
        import csv
        csv_path = out_dir / f"{in_path.stem}.csv"
        fieldnames = list(flat_rows[0].keys()) if flat_rows else [
            "id","case_information","physical_examination","diagnostic_tests","final_diagnosis","option_a","option_b","option_c","option_d","right_option"
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in flat_rows:
                writer.writerow(r)

    # Also write a 100-row sample for quick inspection
    sample_rows = flat_rows[:100]
    sample_jsonl = out_dir / f"{in_path.stem}.sample100.jsonl"
    with sample_jsonl.open("w", encoding="utf-8") as f:
        for r in sample_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Engine used: {engine}")
    print(f"Wrote: {jsonl_path}")
    print(f"Wrote: {out_dir / f'{in_path.stem}.csv'}")
    print(f"Wrote: {sample_jsonl}")


if __name__ == "__main__":
    main()
