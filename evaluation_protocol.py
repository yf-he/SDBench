"""Evaluation protocol implementation for SDBench."""

from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
from data_models import DiagnosticEncounter, BenchmarkResult, JudgeScore
from config import Config

class EvaluationProtocol:
    """Handles evaluation metrics and result analysis for SDBench."""
    
    def __init__(self, config: Config):
        self.config = config
        self.correct_threshold = config.CORRECT_DIAGNOSIS_THRESHOLD
    
    def calculate_diagnostic_accuracy(self, encounters: List[DiagnosticEncounter]) -> float:
        """Calculate diagnostic accuracy based on judge scores."""
        if not encounters:
            return 0.0
        
        correct_cases = 0
        for encounter in encounters:
            if encounter.judge_score and encounter.judge_score.score >= self.correct_threshold:
                correct_cases += 1
        
        return correct_cases / len(encounters)
    
    def calculate_average_cost(self, encounters: List[DiagnosticEncounter]) -> float:
        """Calculate average cumulative cost across all encounters."""
        if not encounters:
            return 0.0
        
        total_cost = sum(encounter.total_cost for encounter in encounters)
        return total_cost / len(encounters)
    
    def evaluate_encounters(self, encounters: List[DiagnosticEncounter]) -> BenchmarkResult:
        """Evaluate a list of diagnostic encounters and return benchmark results."""
        diagnostic_accuracy = self.calculate_diagnostic_accuracy(encounters)
        average_cost = self.calculate_average_cost(encounters)
        
        correct_cases = sum(1 for encounter in encounters 
                           if encounter.judge_score and encounter.judge_score.score >= self.correct_threshold)
        
        return BenchmarkResult(
            diagnostic_accuracy=diagnostic_accuracy,
            average_cost=average_cost,
            total_cases=len(encounters),
            correct_cases=correct_cases,
            encounter_results=encounters
        )
    
    def generate_performance_plot(self, results: List[BenchmarkResult], 
                                agent_names: List[str] = None,
                                save_path: str = None) -> None:
        """Generate a 2D performance plot showing cost vs accuracy."""
        if not results:
            print("No results to plot")
            return
        
        costs = [result.average_cost for result in results]
        accuracies = [result.diagnostic_accuracy for result in results]
        
        plt.figure(figsize=(10, 8))
        plt.scatter(costs, accuracies, s=100, alpha=0.7)
        
        # Add labels for each point if agent names provided
        if agent_names and len(agent_names) == len(results):
            for i, name in enumerate(agent_names):
                plt.annotate(name, (costs[i], accuracies[i]), 
                           xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Average Cost ($)')
        plt.ylabel('Diagnostic Accuracy')
        plt.title('SDBench Performance: Cost vs Accuracy')
        plt.grid(True, alpha=0.3)
        
        # Set axis limits
        plt.xlim(0, max(costs) * 1.1)
        plt.ylim(0, 1.05)
        
        # Add performance regions
        plt.axhspan(0.8, 1.0, alpha=0.1, color='green', label='High Accuracy')
        plt.axhspan(0.6, 0.8, alpha=0.1, color='yellow', label='Medium Accuracy')
        plt.axhspan(0.0, 0.6, alpha=0.1, color='red', label='Low Accuracy')
        
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance plot saved to {save_path}")
        else:
            plt.show()
    
    def calculate_pareto_frontier(self, results: List[BenchmarkResult]) -> List[BenchmarkResult]:
        """Calculate the Pareto frontier for cost vs accuracy trade-offs."""
        if len(results) < 2:
            return results
        
        # Sort by cost (ascending)
        sorted_results = sorted(results, key=lambda x: x.average_cost)
        pareto_frontier = [sorted_results[0]]
        
        for result in sorted_results[1:]:
            # Check if this result dominates the last one in the frontier
            last_frontier = pareto_frontier[-1]
            if (result.diagnostic_accuracy > last_frontier.diagnostic_accuracy or
                (result.diagnostic_accuracy == last_frontier.diagnostic_accuracy and 
                 result.average_cost < last_frontier.average_cost)):
                pareto_frontier.append(result)
        
        return pareto_frontier
    
    def generate_detailed_report(self, result: BenchmarkResult) -> str:
        """Generate a detailed text report of benchmark results."""
        report = f"""
SDBench Evaluation Report
========================

Overall Performance:
- Diagnostic Accuracy: {result.diagnostic_accuracy:.2%} ({result.correct_cases}/{result.total_cases} cases)
- Average Cost: ${result.average_cost:.2f}

Case-by-Case Results:
"""
        
        for i, encounter in enumerate(result.encounter_results, 1):
            report += f"\nCase {i} (ID: {encounter.case_id}):\n"
            report += f"  - Final Diagnosis: {encounter.final_diagnosis or 'No diagnosis'}\n"
            report += f"  - Judge Score: {encounter.judge_score.score if encounter.judge_score else 'N/A'}/5\n"
            report += f"  - Judge Label: {encounter.judge_score.label if encounter.judge_score else 'N/A'}\n"
            report += f"  - Total Cost: ${encounter.total_cost:.2f}\n"
            report += f"  - Number of Actions: {len(encounter.actions)}\n"
            
            if encounter.judge_score and encounter.judge_score.reasoning:
                report += f"  - Judge Reasoning: {encounter.judge_score.reasoning}\n"
        
        return report
    
    def export_results_to_csv(self, results: List[BenchmarkResult], 
                            agent_names: List[str] = None,
                            filename: str = "sdbench_results.csv") -> None:
        """Export results to CSV format for further analysis."""
        import pandas as pd
        
        data = []
        for i, result in enumerate(results):
            agent_name = agent_names[i] if agent_names and i < len(agent_names) else f"Agent_{i+1}"
            
            data.append({
                'Agent': agent_name,
                'Diagnostic_Accuracy': result.diagnostic_accuracy,
                'Average_Cost': result.average_cost,
                'Total_Cases': result.total_cases,
                'Correct_Cases': result.correct_cases
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")
    
    def compare_agents(self, results: List[BenchmarkResult], 
                      agent_names: List[str] = None) -> Dict[str, Any]:
        """Compare multiple agents and return comparison statistics."""
        if not results:
            return {}
        
        comparison = {
            'best_accuracy': max(results, key=lambda x: x.diagnostic_accuracy),
            'lowest_cost': min(results, key=lambda x: x.average_cost),
            'best_efficiency': None,  # Will calculate efficiency score
            'statistics': {}
        }
        
        # Calculate efficiency score (accuracy per dollar)
        for i, result in enumerate(results):
            if result.average_cost > 0:
                efficiency = result.diagnostic_accuracy / result.average_cost
                if comparison['best_efficiency'] is None or efficiency > comparison['best_efficiency'][1]:
                    comparison['best_efficiency'] = (result, efficiency)
        
        # Calculate statistics
        accuracies = [r.diagnostic_accuracy for r in results]
        costs = [r.average_cost for r in results]
        
        comparison['statistics'] = {
            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'cost_mean': np.mean(costs),
            'cost_std': np.std(costs),
            'accuracy_range': (min(accuracies), max(accuracies)),
            'cost_range': (min(costs), max(costs))
        }
        
        return comparison
