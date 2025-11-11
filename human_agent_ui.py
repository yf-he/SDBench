"""
Streamlit UI allowing a human user to take the role of the diagnostic agent.

The human can iteratively submit questions/tests/diagnoses to the gatekeeper,
see responses, track cost, and finally view the judge evaluation.
"""

import json
import os
from pathlib import Path
from typing import List

import streamlit as st

from config import Config
from sdbench import SDBench
from data_loader import CaseFile, load_jsonl_cases
from gatekeeper_agent import GatekeeperAgent
from judge_agent import JudgeAgent
from cost_estimator import CostEstimator
from data_models import ActionType, AgentAction, DiagnosticEncounter, GatekeeperResponse

DEFAULT_DATASET = "/Users/yufei/Desktop/SDBench/converted/test-00000-of-00001.jsonl"


@st.cache_resource
def load_cases(dataset_path: str) -> List[CaseFile]:
    dataset_path = str(Path(dataset_path).expanduser())
    return load_jsonl_cases(dataset_path, publication_year=2025, is_test_case=True)


def initialize_state():
    if "case" not in st.session_state:
        st.session_state.case = None
    if "encounter" not in st.session_state:
        st.session_state.encounter = None
    if "actions" not in st.session_state:
        st.session_state.actions = []
    if "responses" not in st.session_state:
        st.session_state.responses = []
    if "judge_score" not in st.session_state:
        st.session_state.judge_score = None
    if "total_cost" not in st.session_state:
        st.session_state.total_cost = 0.0


def add_action(action: AgentAction, gatekeeper: GatekeeperAgent, case: CaseFile, cost_estimator: CostEstimator):
    st.session_state.actions.append(action)

    if action.action_type == ActionType.DIAGNOSE:
        st.session_state.responses.append(GatekeeperResponse(response_text="Diagnosis submitted."))
        return

    # For questions/tests, call gatekeeper
    response = gatekeeper.process_action(action, case)
    st.session_state.responses.append(response)

    # Update cost for tests
    if action.action_type == ActionType.REQUEST_TESTS:
        test_cost = cost_estimator.calculate_test_cost(action.content)
        st.session_state.total_cost += test_cost
        st.info(f"Estimated test cost: ${test_cost:.2f}")


def finalize_diagnosis(judge: JudgeAgent, case: CaseFile):
    final_action = next((a for a in reversed(st.session_state.actions) if a.action_type == ActionType.DIAGNOSE), None)
    if not final_action:
        st.warning("Please submit a diagnosis before finalizing.")
        return
    st.session_state.judge_score = judge.evaluate_diagnosis(final_action.content, case)


def build_transcript():
    lines = []
    case = st.session_state.case
    judge_score = st.session_state.judge_score

    lines.append("========== Human-Agent Transcript ==========\n")
    lines.append(f"Case ID: {case.case_id}")
    lines.append("Agent: Human")
    lines.append(f"Initial Abstract: {case.initial_abstract}\n")
    for idx, action in enumerate(st.session_state.actions, 1):
        lines.append(f"---------- TURN {idx} ----------")
        lines.append(f"[Human Agent] ({action.action_type.value})")
        lines.append(action.content)
        if idx <= len(st.session_state.responses):
            resp = st.session_state.responses[idx - 1]
            lines.append("[Gatekeeper Response]")
            lines.append(resp.response_text)

    lines.append("\n----------------------------------------")
    lines.append(f"[Cost] Total estimated cost: ${st.session_state.total_cost:.2f}")
    if judge_score:
        lines.append("========================================")
        lines.append("[JUDGE]")
        lines.append(f"Score: {judge_score.score}/5")
        lines.append(f"Label: {judge_score.label}")
        lines.append("Reasoning:")
        lines.append(judge_score.reasoning)

    lines.append("========================================")
    lines.append("[GROUND TRUTH DIAGNOSIS]")
    lines.append(case.ground_truth_diagnosis)
    lines.append("----------------------------------------")
    lines.append("[FULL CASE DATA]")
    lines.append(case.full_case_text)

    return "\n".join(lines)


def main():
    st.set_page_config(page_title="Human Agent Diagnostic UI", layout="wide")
    st.title("Human-in-the-Loop Diagnostic Explorer")

    initialize_state()

    dataset_path = st.text_input("Dataset (.sdbench.jsonl)", value=DEFAULT_DATASET)
    if not dataset_path:
        st.stop()
    try:
        cases = load_cases(dataset_path)
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        st.stop()

    case_ids = [case.case_id for case in cases]
    selected_case = st.selectbox("Select Case ID", case_ids, index=0 if case_ids else None)
    if selected_case:
        st.session_state.case = next(c for c in cases if c.case_id == selected_case)

    cfg = Config()
    gatekeeper = GatekeeperAgent(cfg)
    judge = JudgeAgent(cfg)
    cost_estimator = CostEstimator(cfg)

    if st.button("Reset"):
        for key in ["case", "encounter", "actions", "responses", "judge_score", "total_cost"]:
            st.session_state.pop(key, None)
        st.rerun()

    case = st.session_state.case
    if not case:
        st.stop()

    st.subheader("Case Summary")
    st.write(case.initial_abstract)

    st.subheader("Interaction")
    action_col, submit_col = st.columns([3, 1])
    action_type = action_col.selectbox(
        "Action Type",
        ["ask question", "request test", "diagnose"],
        format_func=lambda x: x.title(),
    )
    content = action_col.text_area("Content", height=150)

    if submit_col.button("Submit Action", use_container_width=True):
        if not content.strip():
            st.warning("Please enter content.")
        else:
            if action_type == "ask question":
                act = AgentAction(action_type=ActionType.ASK_QUESTIONS, content=content.strip())
            elif action_type == "request test":
                act = AgentAction(action_type=ActionType.REQUEST_TESTS, content=content.strip())
            else:
                act = AgentAction(action_type=ActionType.DIAGNOSE, content=content.strip())
            add_action(act, gatekeeper, case, cost_estimator)
            st.rerun()

    if st.session_state.actions:
        st.subheader("Dialogue History")
        for idx, action in enumerate(st.session_state.actions, 1):
            with st.expander(f"Turn {idx}: {action.action_type.value.title()}"):
                st.markdown(f"**Human Agent:**\n\n{action.content}")
                if idx <= len(st.session_state.responses):
                    st.markdown(f"**Gatekeeper:**\n\n{st.session_state.responses[idx-1].response_text}")

    st.markdown(f"**Running Cost Estimate:** ${st.session_state.total_cost:.2f}")

    if st.button("Finalize & Evaluate Diagnosis"):
        finalize_diagnosis(judge, case)
        st.rerun()

    if st.session_state.judge_score:
        st.subheader("Judge Evaluation")
        st.markdown(
            f"""
- **Score:** {st.session_state.judge_score.score}/5
- **Label:** {st.session_state.judge_score.label}
- **Reasoning:**\n\n{st.session_state.judge_score.reasoning}
"""
        )
        st.subheader("Ground Truth Diagnosis")
        st.markdown(f"**{case.ground_truth_diagnosis}**")
        st.subheader("Full Transcript")
        transcript_text = build_transcript()
        st.code(transcript_text, language="markdown")

    st.subheader("Download Transcript")
    transcript_text = build_transcript()
    st.download_button(
        label="Download Transcript",
        data=transcript_text,
        file_name=f"{case.case_id}_HumanAgentTranscript.txt",
        mime="text/plain",
    )


if __name__ == "__main__":
    main()


