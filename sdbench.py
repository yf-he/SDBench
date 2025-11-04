"""Main SDBench implementation - Sequential Diagnosis Benchmark."""

import time
from typing import List, Optional, Callable, Any
from data_models import (
    CaseFile, AgentAction, ActionType, DiagnosticEncounter, 
    BenchmarkResult, GatekeeperResponse
)
from gatekeeper_agent import GatekeeperAgent
from cost_estimator import CostEstimator
from judge_agent import JudgeAgent
from evaluation_protocol import EvaluationProtocol
from config import Config

class DiagnosticAgent:
    """Base class for diagnostic agents to be evaluated."""
    
    def __init__(self, name: str):
        self.name = name
    
    def get_next_action(self, case_abstract: str, encounter_history: List[AgentAction]) -> AgentAction:
        """Get the next action to take in the diagnostic process."""
        raise NotImplementedError("Subclasses must implement get_next_action")
    
    def reset(self) -> None:
        """Reset the agent's state for a new case."""
        pass

class SDBench:
    """Main SDBench class that orchestrates the sequential diagnosis benchmark."""
    
    def __init__(self, config: Config):
        self.config = config
        self.gatekeeper = GatekeeperAgent(config)
        self.cost_estimator = CostEstimator(config)
        self.judge = JudgeAgent(config)
        self.evaluator = EvaluationProtocol(config)
    
    def run_single_encounter(self, diagnostic_agent: DiagnosticAgent,
                           case_file: CaseFile,
                           max_turns: int = 20,
                           disable_cost: bool = False,
                           transcript_dir: str = None) -> DiagnosticEncounter:
        """Run a single diagnostic encounter between agent and case."""
        encounter = DiagnosticEncounter(case_id=case_file.case_id)
        
        # Reset agent for new case
        diagnostic_agent.reset()
        
        # Initialize with case abstract (+ multiple-choice options if available)
        current_context = case_file.initial_abstract
        # Try to parse options from full case text
        try:
            import re
            options_block = []
            in_opts = False
            for line in case_file.full_case_text.splitlines():
                if not in_opts and line.strip().upper().startswith("OPTIONS"):
                    in_opts = True
                    continue
                if in_opts:
                    m = re.match(r"^\s*([ABCD])\s*\.\s*(.+)$", line.strip())
                    if m:
                        options_block.append(f"{m.group(1)}: {m.group(2)}")
            if options_block:
                current_context += "\n\nOptions (choose one):\n" + "\n".join(options_block)
        except Exception:
            pass
        # Build transcript
        transcript_lines = []
        agent_model = getattr(diagnostic_agent, "model", "-")
        gatekeeper_model = getattr(self.gatekeeper, "model", "-")
        judge_model = getattr(self.judge, "model", "-")
        transcript_lines.append(f"========== SDBench Transcript ==========")
        transcript_lines.append(f"Case ID: {case_file.case_id}")
        transcript_lines.append(f"Agent: {diagnostic_agent.name} (model: {agent_model})")
        transcript_lines.append(f"Gatekeeper model: {gatekeeper_model}")
        transcript_lines.append(f"Judge model: {judge_model}")
        transcript_lines.append("----------------------------------------")
        transcript_lines.append("[INITIAL ABSTRACT]")
        transcript_lines.append(case_file.initial_abstract)
        transcript_lines.append("----------------------------------------")
        
        for turn in range(max_turns):
            try:
                # Get next action from diagnostic agent
                action = diagnostic_agent.get_next_action(current_context, encounter.actions)

                # If this is the final turn and agent did not diagnose, force a final diagnosis
                if turn == max_turns - 1 and action.action_type != ActionType.DIAGNOSE:
                    forced = None
                    try:
                        if hasattr(diagnostic_agent, "_make_final_diagnosis"):
                            forced = diagnostic_agent._make_final_diagnosis(case_file.initial_abstract, encounter.actions)
                    except Exception:
                        forced = None
                    try:
                        if forced is None and hasattr(diagnostic_agent, "_force_final_diagnosis_maidxo"):
                            forced = diagnostic_agent._force_final_diagnosis_maidxo(current_context)
                    except Exception:
                        forced = None
                    if isinstance(forced, AgentAction) and forced.action_type == ActionType.DIAGNOSE:
                        action = forced
                    else:
                        action = AgentAction(action_type=ActionType.DIAGNOSE, content="Unable to determine diagnosis with available information.")
                encounter.actions.append(action)
                
                # Process action based on type
                transcript_lines.append(f"---------- TURN {turn+1} ----------")
                transcript_lines.append(f"[Agent: {diagnostic_agent.name} | model: {agent_model}] ({action.action_type.value})")
                transcript_lines.append(action.content)
                if action.action_type == ActionType.DIAGNOSE:
                    # Final diagnosis - end encounter
                    encounter.final_diagnosis = action.content
                    encounter.is_complete = True
                    transcript_lines.append("----------------------------------------")
                    transcript_lines.append("[Final Diagnosis Submitted]")
                    transcript_lines.append(action.content)
                    break
                
                elif action.action_type in [ActionType.ASK_QUESTIONS, ActionType.REQUEST_TESTS]:
                    # Validate request with gatekeeper
                    is_valid, validation_message = self.gatekeeper.validate_request(action)
                    
                    if not is_valid:
                        # Invalid request - provide feedback and continue
                        response = GatekeeperResponse(
                            response_text=f"Invalid request: {validation_message}",
                            is_synthetic=False
                        )
                        encounter.gatekeeper_responses.append(response)
                        transcript_lines.append("[Gatekeeper Response]")
                        transcript_lines.append(response.response_text)
                        continue
                    
                    # Process valid request
                    response = self.gatekeeper.process_action(action, case_file)
                    encounter.gatekeeper_responses.append(response)
                    
                    # Update context with response
                    current_context += f"\n\nResponse: {response.response_text}"
                    transcript_lines.append(f"[Gatekeeper (model: {gatekeeper_model}) Response]")
                    transcript_lines.append(response.response_text)

                    # If agent is MAI-DxO (single-LLM) and has panel_rounds, dump latest debate block
                    try:
                        if hasattr(diagnostic_agent, 'panel_rounds') and diagnostic_agent.panel_rounds:
                            round_idx = len(diagnostic_agent.panel_rounds)
                            transcript_lines.append("----------------------------------------")
                            transcript_lines.append(f"[MAI-DxO Debate Round {round_idx}]")
                            transcript_lines.append(diagnostic_agent.panel_rounds[-1])
                    except Exception:
                        pass

                    # If agent is MultiLLMDxO and has panel_trace of five roles, dump them
                    try:
                        if hasattr(diagnostic_agent, 'panel_trace') and diagnostic_agent.panel_trace:
                            transcript_lines.append("----------------------------------------")
                            transcript_lines.append("[MAI-DxO(5x) Debate Roles]")
                            roles = ["Dr. Hypothesis", "Dr. Test-Chooser", "Dr. Challenger", "Dr. Stewardship", "Dr. Checklist"]
                            for role, content in zip(roles, diagnostic_agent.panel_trace):
                                transcript_lines.append(f"<{role}>")
                                transcript_lines.append(content)
                    except Exception:
                        pass
                    
                    # Calculate cost for this action
                    if not disable_cost:
                        if action.action_type == ActionType.REQUEST_TESTS:
                            test_cost = self.cost_estimator.calculate_test_cost(action.content)
                            encounter.total_cost += test_cost
                            transcript_lines.append(f"[Cost] Estimated test cost: ${test_cost:.2f}")
                
                else:
                    raise ValueError(f"Unknown action type: {action.action_type}")
                
            except Exception as e:
                print(f"Error in turn {turn}: {e}")
                # Add error response and continue
                error_response = GatekeeperResponse(
                    response_text=f"Error processing request: {str(e)}",
                    is_synthetic=False
                )
                encounter.gatekeeper_responses.append(error_response)
                transcript_lines.append("[Gatekeeper Error]")
                transcript_lines.append(str(e))
                continue
        
        # Calculate total visit costs
        if not disable_cost:
            visit_cost = self.cost_estimator.calculate_visit_cost(encounter.actions)
            encounter.total_cost += visit_cost
            transcript_lines.append(f"[Cost] Visit cost total: ${visit_cost:.2f}")
        
        # If encounter completed with diagnosis, evaluate it
        if encounter.is_complete and encounter.final_diagnosis:
            encounter.judge_score = self.judge.evaluate_diagnosis(
                encounter.final_diagnosis, case_file
            )
            transcript_lines.append("========================================")
            transcript_lines.append(f"[JUDGE (model: {judge_model})]")
            transcript_lines.append(f"Score: {encounter.judge_score.score}/5")
            transcript_lines.append(f"Label: {encounter.judge_score.label}")
            transcript_lines.append("Reasoning:")
            transcript_lines.append(encounter.judge_score.reasoning)
        if not disable_cost:
            transcript_lines.append("----------------------------------------")
            transcript_lines.append(f"[Cost] Total estimated cost: ${encounter.total_cost:.2f}")
        
        # Persist transcript if requested
        if transcript_dir:
            import os
            import re
            os.makedirs(transcript_dir, exist_ok=True)
            agent_name_safe = re.sub(r"[^A-Za-z0-9._-]+", "_", diagnostic_agent.name)
            out_path = os.path.join(transcript_dir, f"{case_file.case_id}_{agent_name_safe}.txt")
            try:
                # Append full case and label at the end for completeness
                transcript_lines.append("========================================")
                transcript_lines.append("[FULL CASE DATA]")
                transcript_lines.append(case_file.full_case_text)
                transcript_lines.append("----------------------------------------")
                transcript_lines.append("[GROUND TRUTH DIAGNOSIS]")
                transcript_lines.append(case_file.ground_truth_diagnosis)
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(transcript_lines))
                print(f"Transcript saved: {out_path}")
            except Exception as e:
                print(f"Failed to write transcript: {e}")
        # Also print to stdout
        try:
            print("\n".join(transcript_lines))
        except Exception:
            pass
        return encounter
    
    def run_benchmark(self, diagnostic_agent: DiagnosticAgent,
                     case_files: List[CaseFile],
                     max_turns_per_case: int = 20,
                     disable_cost: bool = False,
                     transcript_dir: str = None) -> BenchmarkResult:
        """Run the full benchmark on a set of cases."""
        print(f"Running SDBench for {diagnostic_agent.name} on {len(case_files)} cases...")
        
        encounters = []
        
        for i, case_file in enumerate(case_files):
            print(f"Processing case {i+1}/{len(case_files)}: {case_file.case_id}")
            
            encounter = self.run_single_encounter(
                diagnostic_agent, case_file, max_turns_per_case, disable_cost=disable_cost, transcript_dir=transcript_dir
            )
            encounters.append(encounter)
            
            # Print encounter summary
            if encounter.is_complete:
                print(f"  ✓ Completed in {len(encounter.actions)} turns")
                print(f"  ✓ Final diagnosis: {encounter.final_diagnosis}")
                print(f"  ✓ Judge score: {encounter.judge_score.score}/5" if encounter.judge_score else "  ✗ No judge score")
                print(f"  ✓ Total cost: ${encounter.total_cost:.2f}")
            else:
                print(f"  ✗ Incomplete (max turns reached)")
                print(f"  ✓ Total cost: ${encounter.total_cost:.2f}")
        
        # Evaluate all encounters
        result = self.evaluator.evaluate_encounters(encounters)
        
        print(f"\nBenchmark Results for {diagnostic_agent.name}:")
        print(f"  Diagnostic Accuracy: {result.diagnostic_accuracy:.2%}")
        print(f"  Average Cost: ${result.average_cost:.2f}")
        print(f"  Correct Cases: {result.correct_cases}/{result.total_cases}")
        
        return result
    
    def run_comparative_benchmark(self, diagnostic_agents: List[DiagnosticAgent],
                                case_files: List[CaseFile],
                                max_turns_per_case: int = 20) -> List[BenchmarkResult]:
        """Run benchmark on multiple agents for comparison."""
        results = []
        
        for agent in diagnostic_agents:
            print(f"\n{'='*50}")
            print(f"Evaluating {agent.name}")
            print(f"{'='*50}")
            
            result = self.run_benchmark(agent, case_files, max_turns_per_case)
            results.append(result)
        
        return results
    
    def generate_performance_report(self, results: List[BenchmarkResult],
                                  agent_names: List[str] = None,
                                  save_plot: str = None) -> None:
        """Generate comprehensive performance report."""
        if not results:
            print("No results to report")
            return
        
        print(f"\n{'='*60}")
        print("SDBench Performance Report")
        print(f"{'='*60}")
        
        # Print individual results
        for i, result in enumerate(results):
            agent_name = agent_names[i] if agent_names and i < len(agent_names) else f"Agent_{i+1}"
            print(f"\n{agent_name}:")
            print(f"  Diagnostic Accuracy: {result.diagnostic_accuracy:.2%}")
            print(f"  Average Cost: ${result.average_cost:.2f}")
            print(f"  Correct Cases: {result.correct_cases}/{result.total_cases}")
        
        # Generate comparison
        if len(results) > 1:
            comparison = self.evaluator.compare_agents(results, agent_names)
            print(f"\nComparison Summary:")
            print(f"  Best Accuracy: {comparison['best_accuracy'].diagnostic_accuracy:.2%}")
            print(f"  Lowest Cost: ${comparison['lowest_cost'].average_cost:.2f}")
            if comparison['best_efficiency']:
                print(f"  Best Efficiency: {comparison['best_efficiency'][1]:.4f} accuracy per $")
        
        # Generate performance plot
        self.evaluator.generate_performance_plot(results, agent_names, save_plot)
    
    def export_results(self, results: List[BenchmarkResult],
                      agent_names: List[str] = None,
                      filename: str = "sdbench_results.csv") -> None:
        """Export results to CSV for further analysis."""
        self.evaluator.export_results_to_csv(results, agent_names, filename)
    
    def run_interactive_demo(self, case_file: CaseFile) -> None:
        """Run an interactive demo where user can manually input actions."""
        print(f"\nInteractive SDBench Demo")
        print(f"Case: {case_file.case_id}")
        print(f"Initial Abstract: {case_file.initial_abstract}")
        print(f"\nAvailable actions:")
        print(f"  - Ask questions: <question>Your question here</question>")
        print(f"  - Request tests: <test>Test name here</test>")
        print(f"  - Make diagnosis: <diagnosis>Your diagnosis here</diagnosis>")
        print(f"\nType 'quit' to exit\n")
        
        encounter = DiagnosticEncounter(case_id=case_file.case_id)
        current_context = case_file.initial_abstract
        
        while True:
            user_input = input("Enter your action: ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            try:
                # Parse action from user input
                action = self._parse_user_action(user_input)
                if not action:
                    print("Invalid action format. Please use the specified XML tags.")
                    continue
                
                encounter.actions.append(action)
                
                if action.action_type == ActionType.DIAGNOSE:
                    encounter.final_diagnosis = action.content
                    encounter.is_complete = True
                    print(f"\nFinal diagnosis: {action.content}")
                    break
                
                elif action.action_type in [ActionType.ASK_QUESTIONS, ActionType.REQUEST_TESTS]:
                    # Process with gatekeeper
                    is_valid, validation_message = self.gatekeeper.validate_request(action)
                    
                    if not is_valid:
                        print(f"Invalid request: {validation_message}")
                        continue
                    
                    response = self.gatekeeper.process_action(action, case_file)
                    encounter.gatekeeper_responses.append(response)
                    
                    print(f"\nResponse: {response.response_text}")
                    
                    # Calculate cost
                    if action.action_type == ActionType.REQUEST_TESTS:
                        test_cost = self.cost_estimator.calculate_test_cost(action.content)
                        encounter.total_cost += test_cost
                        print(f"Test cost: ${test_cost:.2f}")
                
            except Exception as e:
                print(f"Error: {e}")
        
        # Calculate total costs
        visit_cost = self.cost_estimator.calculate_visit_cost(encounter.actions)
        encounter.total_cost += visit_cost
        
        # Evaluate final diagnosis
        if encounter.is_complete and encounter.final_diagnosis:
            encounter.judge_score = self.judge.evaluate_diagnosis(
                encounter.final_diagnosis, case_file
            )
            print(f"\nJudge Score: {encounter.judge_score.score}/5")
            print(f"Judge Reasoning: {encounter.judge_score.reasoning}")
        
        print(f"\nTotal Cost: ${encounter.total_cost:.2f}")
        print(f"Total Actions: {len(encounter.actions)}")
    
    def _parse_user_action(self, user_input: str) -> Optional[AgentAction]:
        """Parse user input into an AgentAction."""
        import re
        
        # Look for question tags
        question_match = re.search(r'<question>(.*?)</question>', user_input, re.DOTALL)
        if question_match:
            return AgentAction(
                action_type=ActionType.ASK_QUESTIONS,
                content=question_match.group(1).strip()
            )
        
        # Look for test tags
        test_match = re.search(r'<test>(.*?)</test>', user_input, re.DOTALL)
        if test_match:
            return AgentAction(
                action_type=ActionType.REQUEST_TESTS,
                content=test_match.group(1).strip()
            )
        
        # Look for diagnosis tags
        diagnosis_match = re.search(r'<diagnosis>(.*?)</diagnosis>', user_input, re.DOTALL)
        if diagnosis_match:
            return AgentAction(
                action_type=ActionType.DIAGNOSE,
                content=diagnosis_match.group(1).strip()
            )
        
        return None
