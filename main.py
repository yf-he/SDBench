"""Main script to run SDBench tests and demonstrations."""

import os
import sys
from typing import List
from config import Config
from sdbench import SDBench
from synthetic_cases import get_all_synthetic_cases
from example_agents import (
    RandomDiagnosticAgent,
    LLMDiagnosticAgent,
    ConservativeDiagnosticAgent,
    AggressiveDiagnosticAgent
)
from data_loader import load_jsonl_cases

def setup_environment():
    """Set up the environment and validate configuration."""
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Validate configuration
    try:
        config = Config()
        config.validate()
        print("✓ Configuration validated successfully")
        return config
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        print("Please set your OPENAI_API_KEY environment variable")
        sys.exit(1)

def run_single_case_demo():
    """Run a demonstration with a single case."""
    print("\n" + "="*60)
    print("SDBench Single Case Demo")
    print("="*60)
    
    config = setup_environment()
    sdbench = SDBench(config)
    
    # Get synthetic cases
    cases = get_all_synthetic_cases()
    test_case = cases[0]  # Use first synthetic case
    
    print(f"Testing with case: {test_case.case_id}")
    print(f"Initial Abstract: {test_case.initial_abstract}")
    
    # Create agents
    agents = [
        RandomDiagnosticAgent("Random"),
        ConservativeDiagnosticAgent("Conservative"),
        AggressiveDiagnosticAgent("Aggressive")
    ]
    
    # Add LLM agent if API key is available
    try:
        llm_agent = LLMDiagnosticAgent("LLM", config)
        agents.append(llm_agent)
    except Exception as e:
        print(f"Note: LLM agent not available: {e}")
    
    # Run benchmark on single case
    results = []
    for agent in agents:
        print(f"\n--- Testing {agent.name} ---")
        result = sdbench.run_benchmark(agent, [test_case], max_turns_per_case=10)
        results.append(result)
    
    # Generate report
    agent_names = [agent.name for agent in agents]
    sdbench.generate_performance_report(results, agent_names, "single_case_demo.png")
    
    return results

def run_full_benchmark():
    """Run the full benchmark on all synthetic cases."""
    print("\n" + "="*60)
    print("SDBench Full Benchmark")
    print("="*60)
    
    config = setup_environment()
    sdbench = SDBench(config)
    
    # Get all synthetic cases
    cases = get_all_synthetic_cases()
    print(f"Running benchmark on {len(cases)} synthetic cases")
    
    # Create agents
    agents = [
        RandomDiagnosticAgent("Random"),
        ConservativeDiagnosticAgent("Conservative"),
        AggressiveDiagnosticAgent("Aggressive")
    ]
    
    # Add LLM agent if API key is available
    try:
        llm_agent = LLMDiagnosticAgent("LLM", config)
        agents.append(llm_agent)
    except Exception as e:
        print(f"Note: LLM agent not available: {e}")
    
    # Run comparative benchmark
    results = sdbench.run_comparative_benchmark(agents, cases, max_turns_per_case=15)
    
    # Generate comprehensive report
    agent_names = [agent.name for agent in agents]
    sdbench.generate_performance_report(results, agent_names, "full_benchmark.png")
    sdbench.export_results(results, agent_names, "full_benchmark_results.csv")
    
    return results

def run_interactive_demo():
    """Run an interactive demo where user can manually input actions."""
    print("\n" + "="*60)
    print("SDBench Interactive Demo")
    print("="*60)
    
    config = setup_environment()
    sdbench = SDBench(config)
    
    # Get synthetic cases
    cases = get_all_synthetic_cases()
    
    print("Available cases:")
    for i, case in enumerate(cases):
        print(f"{i+1}. {case.case_id}: {case.initial_abstract[:100]}...")
    
    try:
        choice = int(input("\nSelect a case (1-{}): ".format(len(cases)))) - 1
        if 0 <= choice < len(cases):
            selected_case = cases[choice]
            sdbench.run_interactive_demo(selected_case)
        else:
            print("Invalid choice")
    except ValueError:
        print("Invalid input")

def run_quick_test():
    """Run a quick test to verify the system works."""
    print("\n" + "="*60)
    print("SDBench Quick Test")
    print("="*60)
    
    config = setup_environment()
    sdbench = SDBench(config)
    
    # Test with a simple case
    cases = get_all_synthetic_cases()
    test_case = cases[0]
    
    # Test with random agent
    agent = RandomDiagnosticAgent("TestAgent")
    result = sdbench.run_benchmark(agent, [test_case], max_turns_per_case=5)
    
    print(f"Quick test completed:")
    print(f"  Accuracy: {result.diagnostic_accuracy:.2%}")
    print(f"  Average Cost: ${result.average_cost:.2f}")
    print(f"  Total Cases: {result.total_cases}")
    
    return result

def run_dataset_benchmark(dataset_path: str, limit: int = 0, use_llm: bool = False):
    """Run benchmark on a dataset loaded from sdbench JSONL."""
    print("\n" + "="*60)
    print("SDBench Dataset Benchmark")
    print("="*60)

    if not os.path.exists(dataset_path):
        print(f"✗ Dataset not found: {dataset_path}")
        sys.exit(1)

    # Environment and bench
    config = Config()
    sdbench = SDBench(config)

    # Load cases
    print(f"Loading dataset: {dataset_path}")
    cases = load_jsonl_cases(dataset_path, publication_year=2025, is_test_case=True, limit=limit)
    print(f"Loaded {len(cases)} cases")

    # Only test LLMAgent as requested
    agents: List = []
    try:
        agents.append(LLMDiagnosticAgent("LLM", config))
    except Exception as e:
        print(f"Note: LLM agent not available: {e}")

    # Run and report
    base_dir = os.path.dirname(dataset_path) or "."
    import datetime
    run_tag = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    # Include agent model in folder name
    agent_model = getattr(LLMDiagnosticAgent("tmp", config), 'model', 'model').replace('/', '_') if True else 'model'
    out_dir = os.path.join(base_dir, f"run_{agent_model}_{run_tag}")
    os.makedirs(out_dir, exist_ok=True)
    transcripts_dir = os.path.join(out_dir, "transcripts")
    results = []
    for agent in agents:
        result = sdbench.run_benchmark(
            agent,
            cases,
            max_turns_per_case=15,
            disable_cost=False,
            transcript_dir=transcripts_dir,
        )
        results.append(result)
    agent_names = [a.name for a in agents]
    plot_path = os.path.join(out_dir, "dataset_benchmark.png")
    csv_path = os.path.join(out_dir, "dataset_benchmark.csv")

    sdbench.generate_performance_report(results, agent_names, plot_path)
    sdbench.export_results(results, agent_names, csv_path)

    # Write run summary (accuracy, average score, average cost)
    try:
        import statistics
        total_cases = sum(r.total_cases for r in results)
        correct_cases = sum(r.correct_cases for r in results)
        accuracy = (correct_cases / total_cases) if total_cases else 0.0
        # Average score over all encounters
        all_scores = []
        all_costs = []
        for r in results:
            for enc in r.encounter_results:
                if enc.judge_score:
                    all_scores.append(enc.judge_score.score)
                all_costs.append(enc.total_cost)
        avg_score = statistics.mean(all_scores) if all_scores else 0.0
        avg_cost = statistics.mean(all_costs) if all_costs else 0.0
        with open(os.path.join(out_dir, "run_summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"Agent: {agent_names[0]}\n")
            f.write(f"Agent model: {agent_model}\n")
            f.write(f"Total cases: {total_cases}\n")
            f.write(f"Correct cases: {correct_cases}\n")
            f.write(f"Accuracy: {accuracy:.2%}\n")
            f.write(f"Average judge score: {avg_score:.2f}\n")
            f.write(f"Average estimated cost: ${avg_cost:.2f}\n")
    except Exception as e:
        print(f"Failed to write run summary: {e}")
    return results

def main():
    """Main function to run SDBench demonstrations."""
    print("SDBench - Sequential Diagnosis Benchmark")
    print("========================================")
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        print("\nAvailable modes:")
        print("1. quick        - Run a quick test")
        print("2. single       - Run single case demo")
        print("3. full         - Run full benchmark (synthetic cases)")
        print("4. interactive  - Run interactive demo")
        print("5. all          - Run all synthetic modes")
        print("6. dataset      - Run on dataset JSONL (path in argv[2], optional limit argv[3])")
        
        mode = input("\nSelect mode (1-6): ").strip()
        mode_map = {
            "1": "quick",
            "2": "single", 
            "3": "full",
            "4": "interactive",
            "5": "all",
            "6": "dataset",
        }
        mode = mode_map.get(mode, "quick")
    
    try:
        if mode == "quick":
            run_quick_test()
        elif mode == "single":
            run_single_case_demo()
        elif mode == "full":
            run_full_benchmark()
        elif mode == "interactive":
            run_interactive_demo()
        elif mode == "all":
            run_quick_test()
            run_single_case_demo()
            run_full_benchmark()
        elif mode == "dataset":
            # argv: main.py dataset <path> [limit] [use_llm]
            dataset_path = sys.argv[2] if len(sys.argv) > 2 else \
                "/Users/yufei/Desktop/SDBench/converted/test-00000-of-00001.sdbench.jsonl"
            limit = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 0
            use_llm = False
            if len(sys.argv) > 4:
                use_llm = sys.argv[4].lower() in ("1", "true", "yes", "y")
            run_dataset_benchmark(dataset_path, limit=limit, use_llm=use_llm)
        else:
            print(f"Unknown mode: {mode}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nError running benchmark: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
