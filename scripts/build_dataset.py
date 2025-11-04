import argparse
from pathlib import Path
from data_loader import load_jsonl_cases, save_cases_as_jsonl, save_cases_as_csv


def main():
    parser = argparse.ArgumentParser(description="Build SDBench-ready dataset from converted JSONL")
    parser.add_argument("input", type=str, help="Path to converted JSONL (flattened Options)")
    parser.add_argument("--outdir", type=str, default="converted", help="Output directory")
    parser.add_argument("--publication_year", type=int, default=2025, help="Publication year to annotate cases")
    parser.add_argument("--is_test_case", action="store_true", help="Mark cases as test set")
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit (0 = all)")
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.outdir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = load_jsonl_cases(
        jsonl_path=str(in_path),
        publication_year=args.publication_year,
        is_test_case=args.is_test_case,
        limit=args.limit,
        case_id_prefix="DA_",
    )

    jsonl_out = out_dir / f"{in_path.stem}.sdbench.jsonl"
    csv_out = out_dir / f"{in_path.stem}.sdbench.csv"

    save_cases_as_jsonl(cases, str(jsonl_out))
    try:
        save_cases_as_csv(cases, str(csv_out))
    except Exception as e:
        print(f"CSV export skipped (pandas missing): {e}")

    print(f"Built {len(cases)} cases")
    print(f"JSONL: {jsonl_out}")
    if csv_out.exists():
        print(f"CSV:    {csv_out}")


if __name__ == "__main__":
    main()
