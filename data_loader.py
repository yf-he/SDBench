import json
from pathlib import Path
from typing import Iterable, List, Optional

from data_models import CaseFile


def _first_n_sentences(text: str, max_sentences: int = 2, max_chars: int = 360) -> str:
    if not text:
        return ""
    # Simple sentence split on period/question/exclamation; keep it robust for long paragraphs
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    abstract = " ".join(sentences[:max_sentences]).strip()
    if len(abstract) > max_chars:
        abstract = abstract[: max_chars - 1].rstrip() + "…"
    return abstract


def _build_full_case_text(case_information: Optional[str],
                          physical_examination: Optional[str],
                          diagnostic_tests: Optional[str],
                          option_a: Optional[str],
                          option_b: Optional[str],
                          option_c: Optional[str],
                          option_d: Optional[str]) -> str:
    sections = []
    if case_information:
        sections.append("PRESENTATION OF CASE\n\n" + case_information.strip())
    if physical_examination:
        sections.append("PHYSICAL EXAMINATION\n\n" + physical_examination.strip())
    if diagnostic_tests:
        sections.append("DIAGNOSTIC TESTS\n\n" + diagnostic_tests.strip())
    options_lines = []
    if any([option_a, option_b, option_c, option_d]):
        options_lines.append("OPTIONS")
        if option_a:
            options_lines.append(f"A. {option_a}")
        if option_b:
            options_lines.append(f"B. {option_b}")
        if option_c:
            options_lines.append(f"C. {option_c}")
        if option_d:
            options_lines.append(f"D. {option_d}")
    if options_lines:
        sections.append("\n".join(options_lines))
    return "\n\n".join(sections)


def load_jsonl_cases(jsonl_path: str,
                     publication_year: int = 2025,
                     is_test_case: bool = False,
                     limit: int = 0,
                     case_id_prefix: str = "DA_") -> List[CaseFile]:
    """
    Load converted JSONL and adapt each row to CaseFile for SDBench.

    Expected JSONL fields per row:
      - id (int or str)
      - case_information (str)
      - physical_examination (str)
      - diagnostic_tests (str)
      - final_diagnosis (str)
      - option_a/option_b/option_c/option_d (str, optional)
      - right_option (str, optional)
    """
    path = Path(jsonl_path).expanduser().resolve()
    cases: List[CaseFile] = []

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            rid = row.get("id")
            # Build case_id: prefix + original id, fallback to line number
            case_id = f"{case_id_prefix}{rid}" if rid is not None else f"{case_id_prefix}{i}"

            case_information = row.get("case_information") or ""
            physical_examination = row.get("physical_examination") or ""
            diagnostic_tests = row.get("diagnostic_tests") or ""
            final_dx = row.get("final_diagnosis") or ""

            initial_abstract = _first_n_sentences(case_information or physical_examination or diagnostic_tests)
            full_case_text = _build_full_case_text(
                case_information=case_information,
                physical_examination=physical_examination,
                diagnostic_tests=diagnostic_tests,
                option_a=row.get("option_a"),
                option_b=row.get("option_b"),
                option_c=row.get("option_c"),
                option_d=row.get("option_d"),
            )

            cases.append(CaseFile(
                case_id=case_id,
                initial_abstract=initial_abstract,
                full_case_text=full_case_text,
                ground_truth_diagnosis=final_dx,
                publication_year=publication_year,
                is_test_case=is_test_case,
            ))

            if limit and len(cases) >= limit:
                break

    return cases


def save_cases_as_jsonl(cases: List[CaseFile], out_path: str) -> None:
    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(c.model_dump_json(ensure_ascii=False) + "\n")


def save_cases_as_csv(cases: List[CaseFile], out_path: str) -> None:
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError("pandas is required to export CSV") from e
    import pandas as pd  # for type checkers

    df = pd.DataFrame([
        {
            "case_id": c.case_id,
            "initial_abstract": c.initial_abstract,
            "ground_truth_diagnosis": c.ground_truth_diagnosis,
            "publication_year": c.publication_year,
            "is_test_case": c.is_test_case,
            "full_case_text": c.full_case_text,
        }
        for c in cases
    ])
    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
