import os
import argparse
import datetime
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt

from config import Config
from sdbench import SDBench
from data_loader import load_jsonl_cases
from example_agents import LLMDiagnosticAgent, MultiLLMDxOAgent


def run_for_model(config: Config, dataset_path: str, model_id: str, limit: int = 30) -> Tuple[dict, dict]:
    """Run LLMAgent and MAI-DxO(5xLLM same model) on the same dataset slice.

    Returns (llm_summary, maidxo_summary).
    Each agent's transcripts are saved in separate folders inside per-model run directory.
    """
    cases = load_jsonl_cases(dataset_path, publication_year=2025, is_test_case=True, limit=limit)
    bench = SDBench(config)

    # Per-model output root
    base_dir = os.path.dirname(dataset_path) or "."
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = model_id.replace('/', '_').replace(':', '_')
    run_root = os.path.join(base_dir, f"compare_{model_safe}_{ts}")
    os.makedirs(run_root, exist_ok=True)

    # 1) LLMAgent (single-model) with its own transcripts folder
    llm_agent = LLMDiagnosticAgent(name=f"LLM({model_id})", config=config)
    llm_agent.model = model_id  # override only agent model
    llm_name_safe = llm_agent.name.replace('/', '_').replace(':', '_').replace(' ', '_').replace('(', '_').replace(')', '_')
    llm_transcripts = os.path.join(run_root, f"transcripts_{llm_name_safe}")
    llm_result = bench.run_benchmark(
        llm_agent,
        cases,
        max_turns_per_case=20,
        disable_cost=False,
        transcript_dir=llm_transcripts,
    )
    llm_summary = summarize_result(llm_result, agent_name=llm_agent.name, agent_model=model_id)
    write_summary(os.path.join(run_root, f"summary_{llm_name_safe}.txt"), llm_summary)

    # 2) MAI-DxO(5xLLM same model) with its own transcripts folder
    maidxo_agent = MultiLLMDxOAgent(name=f"MAI-DxO(5x:{model_id})", config=config, model_for_all=model_id)
    maidxo_name_safe = maidxo_agent.name.replace('/', '_').replace(':', '_').replace(' ', '_').replace('(', '_').replace(')', '_')
    maidxo_transcripts = os.path.join(run_root, f"transcripts_{maidxo_name_safe}")
    maidxo_result = bench.run_benchmark(
        maidxo_agent,
        cases,
        max_turns_per_case=20,
        disable_cost=False,
        transcript_dir=maidxo_transcripts,
    )
    maidxo_summary = summarize_result(maidxo_result, agent_name=maidxo_agent.name, agent_model=model_id)
    write_summary(os.path.join(run_root, f"summary_{maidxo_name_safe}.txt"), maidxo_summary)

    # Combined summary CSV for this model
    df = pd.DataFrame([llm_summary, maidxo_summary])
    df.to_csv(os.path.join(run_root, "summary.csv"), index=False)

    # Per-model plot: cost vs accuracy with labels
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df["avg_cost"], df["accuracy"], s=120)
    for _, row in df.iterrows():
        ax.annotate(row["agent"], (row["avg_cost"], row["accuracy"]), xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel("Average Estimated Cost ($)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Performance - {model_id}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(run_root, "compare_plot.png"), dpi=220)
    plt.close(fig)

    return llm_summary, maidxo_summary


def summarize_result(result, agent_name: str, agent_model: str) -> dict:
    accuracy = result.diagnostic_accuracy
    scores = [enc.judge_score.score for enc in result.encounter_results if enc.judge_score]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    costs = [enc.total_cost for enc in result.encounter_results]
    avg_cost = sum(costs) / len(costs) if costs else 0.0
    return {
        "agent": agent_name,
        "model": agent_model,
        "accuracy": accuracy,
        "avg_score": avg_score,
        "avg_cost": avg_cost,
        "total_cases": result.total_cases,
        "correct_cases": result.correct_cases,
    }


def write_summary(path: str, summary: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Agent: {summary['agent']}\n")
        f.write(f"Model: {summary['model']}\n")
        f.write(f"Total cases: {summary['total_cases']}\n")
        f.write(f"Correct cases: {summary['correct_cases']}\n")
        f.write(f"Accuracy: {summary['accuracy']:.2%}\n")
        f.write(f"Average judge score: {summary['avg_score']:.2f}\n")
        f.write(f"Average estimated cost: ${summary['avg_cost']:.2f}\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark 5 LLM agents and 5x-LLM MAI-DxO for each model")
    parser.add_argument("dataset", type=str, help="Path to sdbench JSONL dataset")
    parser.add_argument("--models", type=str, nargs="*", default=None,
                        help="List of 5 model ids (agent-side only)")
    parser.add_argument("--limit", type=int, default=30, help="Number of cases to evaluate")
    args = parser.parse_args()

    config = Config()

    if not args.models or len(args.models) == 0:
        base = config.GATEKEEPER_MODEL
        models = [base, base, base, base, base]
    else:
        models = args.models
        if len(models) != 5:
            print("Please provide exactly 5 model ids or omit --models to use the default repeated 5 times.")
            return

    all_rows = []
    for m in models:
        llm_row, maidxo_row = run_for_model(config, args.dataset, m, limit=args.limit)
        all_rows.append({"kind": "LLM", **llm_row})
        all_rows.append({"kind": "MAI-DxO(5x)", **maidxo_row})

    # Overall summary
    base_dir = os.path.dirname(args.dataset) or "."
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(base_dir, f"overall_compare_{ts}")
    os.makedirs(out_root, exist_ok=True)

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(os.path.join(out_root, "overall_summary.csv"), index=False)

    # Diagram: cost vs accuracy
    fig, ax = plt.subplots(figsize=(8, 6))
    for kind, g in df_all.groupby("kind"):
        ax.scatter(g["avg_cost"], g["accuracy"], s=140, label=kind)
        for _, row in g.iterrows():
            ax.annotate(f"{row['model']}", (row["avg_cost"], row["accuracy"]), xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.set_xlabel("Average Estimated Cost ($)")
    ax.set_ylabel("Accuracy")
    ax.set_title("LLM vs MAI-DxO Performance (Cost vs Accuracy)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_root, "overall_compare.png"), dpi=220)
    plt.close(fig)

    print(f"Wrote overall summary to: {out_root}")


if __name__ == "__main__":
    main()

import os
import argparse
import datetime
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt

from config import Config
from sdbench import SDBench
from data_loader import load_jsonl_cases
from example_agents import LLMDiagnosticAgent, MultiLLMDxOAgent


def run_for_model(config: Config, dataset_path: str, model_id: str, limit: int = 30) -> Tuple[dict, dict]:
    """Run LLMAgent and MAI-DxO(5xLLM same model) on the same dataset slice.

    Returns (llm_summary, maidxo_summary) where each summary is a dict with metrics.
    """
    # Prepare cases
    cases = load_jsonl_cases(dataset_path, publication_year=2025, is_test_case=True, limit=limit)
    bench = SDBench(config)

    # Out directory per model
    base_dir = os.path.dirname(dataset_path) or "."
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(base_dir, f"compare_{model_id.replace('/', '_')}_{ts}")
    os.makedirs(run_root, exist_ok=True)

    # 1) LLMAgent (single-model)
    llm_agent = LLMDiagnosticAgent(name=f"LLM({model_id})", config=config)
    # override model for this run
    llm_agent.model = model_id
    llm_out = os.path.join(run_root, "llm_transcripts")
    llm_result = bench.run_benchmark(
        llm_agent,
        cases,
        max_turns_per_case=20,
        disable_cost=False,
        transcript_dir=llm_out,
    )

    llm_summary = summarize_result(llm_result, agent_name=llm_agent.name, agent_model=model_id)
    write_summary(os.path.join(run_root, "llm_summary.txt"), llm_summary)

    # 2) MAI-DxO(5xLLM same model)
    maidxo_agent = MultiLLMDxOAgent(name=f"MAI-DxO(5x:{model_id})", config=config, model_for_all=model_id)
    maidxo_out = os.path.join(run_root, "maidxo_transcripts")
    maidxo_result = bench.run_benchmark(
        maidxo_agent,
        cases,
        max_turns_per_case=20,
        disable_cost=False,
        transcript_dir=maidxo_out,
    )

    maidxo_summary = summarize_result(maidxo_result, agent_name=maidxo_agent.name, agent_model=model_id)
    write_summary(os.path.join(run_root, "maidxo_summary.txt"), maidxo_summary)

    # Save combined CSV
    df = pd.DataFrame([llm_summary, maidxo_summary])
    df.to_csv(os.path.join(run_root, "summary.csv"), index=False)

    # Plot (cost vs accuracy)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df["avg_cost"], df["accuracy"], s=120)
    for _, row in df.iterrows():
        ax.annotate(row["agent"], (row["avg_cost"], row["accuracy"]), xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel("Average Estimated Cost ($)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Performance - {model_id}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(run_root, "compare_plot.png"), dpi=200)
    plt.close(fig)

    return llm_summary, maidxo_summary


def summarize_result(result, agent_name: str, agent_model: str) -> dict:
    # Diagnostic accuracy already computed in result
    accuracy = result.diagnostic_accuracy
    # Average judge score
    scores = [enc.judge_score.score for enc in result.encounter_results if enc.judge_score]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    # Average cost
    costs = [enc.total_cost for enc in result.encounter_results]
    avg_cost = sum(costs) / len(costs) if costs else 0.0
    return {
        "agent": agent_name,
        "model": agent_model,
        "accuracy": accuracy,
        "avg_score": avg_score,
        "avg_cost": avg_cost,
        "total_cases": result.total_cases,
        "correct_cases": result.correct_cases,
    }


def write_summary(path: str, summary: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Agent: {summary['agent']}\n")
        f.write(f"Model: {summary['model']}\n")
        f.write(f"Total cases: {summary['total_cases']}\n")
        f.write(f"Correct cases: {summary['correct_cases']}\n")
        f.write(f"Accuracy: {summary['accuracy']:.2%}\n")
        f.write(f"Average judge score: {summary['avg_score']:.2f}\n")
        f.write(f"Average estimated cost: ${summary['avg_cost']:.2f}\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark 5 LLM agents and 5x-LLM MAI-DxO for each model")
    parser.add_argument("dataset", type=str, help="Path to sdbench JSONL dataset")
    parser.add_argument("--models", type=str, nargs="*", default=None,
                        help="List of model ids (default: uses config gatekeeper model repeated 5 times)")
    parser.add_argument("--limit", type=int, default=30, help="Number of cases to evaluate")
    args = parser.parse_args()

    config = Config()

    # Five models: if not provided, default to repeating the configured model
    if not args.models or len(args.models) == 0:
        base = config.GATEKEEPER_MODEL
        models = [base, base, base, base, base]
    else:
        models = args.models
        if len(models) != 5:
            print("Please provide exactly 5 model ids or omit --models to use the default repeated 5 times.")
            return

    all_rows = []
    for m in models:
        llm_row, maidxo_row = run_for_model(config, args.dataset, m, limit=args.limit)
        all_rows.append({"kind": "LLM", **llm_row})
        all_rows.append({"kind": "MAI-DxO(5x)", **maidxo_row})

    # Aggregate into one CSV and one plot
    base_dir = os.path.dirname(args.dataset) or "."
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(base_dir, f"overall_compare_{ts}")
    os.makedirs(out_root, exist_ok=True)

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(os.path.join(out_root, "overall_summary.csv"), index=False)

    # Diagram: cost vs accuracy, different markers for LLM vs MAI-DxO
    fig, ax = plt.subplots(figsize=(8, 6))
    for kind, g in df_all.groupby("kind"):
        ax.scatter(g["avg_cost"], g["accuracy"], s=140, label=kind)
        for _, row in g.iterrows():
            ax.annotate(f"{row['model']}", (row["avg_cost"], row["accuracy"]), xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.set_xlabel("Average Estimated Cost ($)")
    ax.set_ylabel("Accuracy")
    ax.set_title("LLM vs MAI-DxO Performance (Cost vs Accuracy)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_root, "overall_compare.png"), dpi=220)
    plt.close(fig)

    print(f"Wrote overall summary to: {out_root}")


if __name__ == "__main__":
    main()


