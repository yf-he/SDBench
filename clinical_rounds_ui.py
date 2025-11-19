"""
Multi-turn Medical Diagnosis Copilot

This interactive demo simulates a real-world clinical diagnostic scenario where:
- You play the role of an attending physician making diagnostic decisions
- An AI resident doctor (LLM) can assist by drafting questions, test orders, or diagnoses
- The Medical Evidence System provides patient information, test results, and clinical data
- Each step builds toward a final diagnosis that is evaluated against ground truth

The scenario mimics a hospital bedside encounter where physicians iteratively:
1. Ask questions about patient history and symptoms
2. Order diagnostic tests and imaging studies
3. Synthesize information to reach a diagnosis

You can choose to draft each step yourself or let the AI resident doctor suggest the next action,
creating a collaborative human-AI diagnostic workflow.
"""

from pathlib import Path
from typing import List, Optional

import streamlit as st

from config import Config
from data_loader import CaseFile, load_jsonl_cases
from gatekeeper_agent import GatekeeperAgent
from judge_agent import JudgeAgent
from cost_estimator import CostEstimator
from data_models import ActionType, AgentAction, GatekeeperResponse
from utils.llm_client import chat_completion_with_retries

DEFAULT_DATASET = Path(
    "C:/Users/t-yufeihe/Downloads/SDBench-main/SDBench-main/converted/converted/test-00000-of-00001.jsonl"
)


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
    if "action_authors" not in st.session_state:
        st.session_state.action_authors = []
    if "judge_score" not in st.session_state:
        st.session_state.judge_score = None
    if "total_cost" not in st.session_state:
        st.session_state.total_cost = 0.0
    if "action_content_input" not in st.session_state:
        st.session_state.action_content_input = ""
    if "action_content_pending" not in st.session_state:
        st.session_state.action_content_pending = None
    if "llm_status" not in st.session_state:
        st.session_state.llm_status = ""


def add_action(
    action: AgentAction,
    gatekeeper: GatekeeperAgent,
    case: CaseFile,
    cost_estimator: CostEstimator,
    author_label: str,
):
    st.session_state.actions.append(action)
    st.session_state.action_authors.append(author_label)

    if action.action_type == ActionType.DIAGNOSE:
        st.session_state.responses.append(
            GatekeeperResponse(response_text="Diagnosis submitted for evaluation.")
        )
        return

    response = gatekeeper.process_action(action, case)
    st.session_state.responses.append(response)

    if action.action_type == ActionType.REQUEST_TESTS:
        test_cost = cost_estimator.calculate_test_cost(action.content)
        st.session_state.total_cost += test_cost
        st.info(f"Estimated test cost: ${test_cost:.2f}")


def finalize_diagnosis(judge: JudgeAgent, case: CaseFile):
    final_action = next(
        (a for a in reversed(st.session_state.actions) if a.action_type == ActionType.DIAGNOSE),
        None,
    )
    if not final_action:
        st.warning("Submit a diagnosis before finalizing rounds.")
        return
    st.session_state.judge_score = judge.evaluate_diagnosis(final_action.content, case)


def build_clinical_context(case: CaseFile) -> str:
    lines = [
        f"Patient ID: {case.case_id}",
        f"Chief Concern Summary: {case.initial_abstract}",
        "",
        "Running Encounter:",
    ]
    for idx, action in enumerate(st.session_state.actions, 1):
        author = st.session_state.action_authors[idx - 1] if idx - 1 < len(st.session_state.action_authors) else "Attending"
        lines.append(f"{idx}. {author} [{action.action_type.value}]: {action.content}")
        if idx - 1 < len(st.session_state.responses):
            resp = st.session_state.responses[idx - 1]
            lines.append(f"   Medical Evidence System: {resp.response_text}")
    return "\n".join(lines)


def suggest_action_with_llm(
    desired_action: ActionType,
    case: CaseFile,
    cfg: Config,
    model_name: str,
    temperature: float = 0.3,
) -> Optional[str]:
    if not case:
        return None

    client = cfg.get_openai_client()
    context = build_clinical_context(case)

    task_map = {
        ActionType.ASK_QUESTIONS: (
            "Draft a single, clinically precise bedside question for the patient or staff "
            "that would meaningfully advance the diagnostic work-up."
        ),
        ActionType.REQUEST_TESTS: (
            "Order exactly one high-yield diagnostic test or imaging study. Include modality and any pertinent qualifiers."
        ),
        ActionType.DIAGNOSE: (
            "Provide a concise, definitive diagnosis (or leading impression) that best explains the presentation."
        ),
    }

    system_prompt = (
        "You are an AI resident doctor assisting an attending physician in clinical diagnosis. "
        "Your role is to collaborate with the attending to manage an undifferentiated patient. "
        "Speak in precise clinical language, avoid revealing hidden ground truth, "
        "and keep outputs to a single actionable sentence."
    )

    user_prompt = f"""
{context}

Task: {task_map[desired_action]}
Respond with only the requested utterance, no preamble.
"""

    try:
        completion = chat_completion_with_retries(
            client=client,
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.strip()},
            ],
            temperature=temperature,
            max_tokens=220,
        )
        content = completion.choices[0].message.content.strip()
        return content
    except Exception as exc:
        st.error(f"AI resident could not draft a response: {exc}")
        return None


def build_transcript(case: CaseFile) -> str:
    lines = []
    judge_score = st.session_state.judge_score

    lines.append("========== Multi-turn Medical Diagnosis Copilot - Clinical Encounter Transcript ==========\n")
    lines.append(f"Patient ID: {case.case_id}")
    lines.append("Care Team: Attending Physician + AI Resident Doctor")
    lines.append(f"Chief Concern: {case.initial_abstract}\n")

    for idx, action in enumerate(st.session_state.actions, 1):
        author = st.session_state.action_authors[idx - 1] if idx - 1 < len(st.session_state.action_authors) else "Attending"
        lines.append(f"---------- ROUND {idx} ----------")
        lines.append(f"[{author}] ({action.action_type.value})")
        lines.append(action.content)
        if idx <= len(st.session_state.responses):
            resp = st.session_state.responses[idx - 1]
            lines.append("[Medical Evidence System]")
            lines.append(resp.response_text)

    lines.append("\n----------------------------------------")
    lines.append(f"[Resource Stewardship] Total estimated cost: ${st.session_state.total_cost:.2f}")

    if judge_score:
        lines.append("========================================")
        lines.append("[ATTENDING BOARD REVIEW]")
        lines.append(f"Score: {judge_score.score}/5")
        lines.append(f"Label: {judge_score.label}")
        lines.append("Reasoning:")
        lines.append(judge_score.reasoning)

    lines.append("========================================")
    lines.append("[REFERENCE DIAGNOSIS]")
    lines.append(case.ground_truth_diagnosis)
    lines.append("----------------------------------------")
    lines.append("[FULL CASE DATA]")
    lines.append(case.full_case_text)
    return "\n".join(lines)


def main():
    st.set_page_config(page_title="Multi-turn Medical Diagnosis Copilot", layout="wide")
    st.title("Multi-turn Medical Diagnosis Copilot")
    
    with st.expander("📋 About This Demo", expanded=True):
        st.markdown("""
        **Simulated Clinical Scenario:**
        
        This interactive demo simulates a real-world medical diagnostic encounter. You play the role of an **attending physician** 
        working through a complex diagnostic case. The scenario includes:
        
        - **You (Attending Physician)**: Make diagnostic decisions, ask questions, order tests, and formulate diagnoses
        - **AI Resident Doctor**: An AI assistant that can draft questions, test orders, or diagnostic impressions on demand
        - **Medical Evidence System**: Provides patient information, answers clinical questions, and returns test results
        - **Evaluation Panel**: Reviews your final diagnosis against ground truth
        
        **How It Works:**
        
        1. **Select a Patient Case**: Choose from available clinical cases
        2. **Review Initial Presentation**: Read the patient's chief complaint and initial findings
        3. **Plan Next Steps**: For each turn, you can:
           - Ask questions about patient history, symptoms, or examination findings
           - Order diagnostic tests or imaging studies
           - Submit a final diagnosis
        4. **Choose Your Approach**: For each action, decide whether to:
           - Draft it yourself (as the attending physician)
           - Let the AI resident doctor suggest a draft (which you can edit before submitting)
        5. **Receive Evidence**: The Medical Evidence System responds with relevant clinical information
        6. **Finalize & Evaluate**: Submit your diagnosis to receive expert evaluation
        
        This creates a collaborative human-AI workflow that mirrors real clinical decision-making processes.
        """)
    
    st.divider()

    initialize_state()

    with st.sidebar:
        st.header("Clinical Setup")
        dataset_path = st.text_input("Dataset (.sdbench.jsonl)", value=str(DEFAULT_DATASET))
        ai_model_id = st.text_input("AI Resident Doctor Model", value="openai/gpt-4o-mini")
        if st.button("Reset Encounter"):
            for key in [
                "case",
                "encounter",
                "actions",
                "responses",
                "action_authors",
                "judge_score",
                "total_cost",
                "action_content_input",
                "action_content_pending",
                "llm_status",
            ]:
                st.session_state.pop(key, None)
            st.rerun()

    if not dataset_path:
        st.stop()

    try:
        cases = load_cases(dataset_path)
    except Exception as exc:
        st.error(f"Failed to load dataset: {exc}")
        st.stop()

    case_ids = [case.case_id for case in cases]
    selected_case = st.selectbox("Select patient chart", case_ids, index=0 if case_ids else None)
    if selected_case:
        st.session_state.case = next(c for c in cases if c.case_id == selected_case)

    cfg = Config()
    gatekeeper = GatekeeperAgent(cfg)
    judge = JudgeAgent(cfg)
    cost_estimator = CostEstimator(cfg)

    case = st.session_state.case
    if not case:
        st.info("Select a case to begin rounds.")
        st.stop()

    st.subheader("Patient Snapshot")
    st.write(case.initial_abstract)

    if st.session_state.action_content_pending is not None:
        st.session_state.action_content_input = st.session_state.action_content_pending
        st.session_state.action_content_pending = None

    st.subheader("Plan the Next Step")
    action_col, submit_col = st.columns([4, 1])
    action_type_label = action_col.selectbox(
        "Clinical action",
        ["ask question", "request test", "diagnose"],
        format_func=lambda x: x.title(),
    )
    actor_label = action_col.radio(
        "Who drafts this step?",
        ["Attending (You)", "AI Resident Doctor (LLM)"],
        horizontal=True,
    )

    action_type_map = {
        "ask question": ActionType.ASK_QUESTIONS,
        "request test": ActionType.REQUEST_TESTS,
        "diagnose": ActionType.DIAGNOSE,
    }
    desired_action = action_type_map[action_type_label]

    if actor_label == "AI Resident Doctor (LLM)":
        if action_col.button("Let AI Resident Doctor draft", use_container_width=False):
            with st.spinner("AI Resident Doctor drafting..."):
                suggestion = suggest_action_with_llm(desired_action, case, cfg, ai_model_id)
            if suggestion:
                st.session_state.action_content_pending = suggestion
                st.session_state.llm_status = f"Drafted by AI Resident Doctor ({ai_model_id})"
                st.rerun()

    content = action_col.text_area(
        "Document your bedside question/order/impression",
        key="action_content_input",
        height=180,
    )

    if st.session_state.llm_status:
        st.info(st.session_state.llm_status)

    if submit_col.button("Submit to Medical Evidence System", use_container_width=True):
        final_content = st.session_state.action_content_input.strip()
        if not final_content:
            st.warning("Enter or generate content before submitting.")
        else:
            author = "AI Resident Doctor" if actor_label == "AI Resident Doctor (LLM)" else "Attending"
            action = AgentAction(action_type=desired_action, content=final_content)
            add_action(action, gatekeeper, case, cost_estimator, author)
            st.session_state.action_content_pending = ""
            st.session_state.llm_status = ""
            st.rerun()

    if st.session_state.actions:
        st.subheader("Clinical Timeline")
        for idx, action in enumerate(st.session_state.actions, 1):
            author = st.session_state.action_authors[idx - 1]
            with st.expander(f"Round {idx}: {author} • {action.action_type.value.replace('_', ' ').title()}"):
                st.markdown(f"**{author}:** {action.content}")
                if idx <= len(st.session_state.responses):
                    st.markdown(f"**Medical Evidence System:** {st.session_state.responses[idx - 1].response_text}")

    st.markdown(f"**Cumulative Estimated Cost:** ${st.session_state.total_cost:.2f}")

    if st.button("Finalize diagnosis & request attending review"):
        finalize_diagnosis(judge, case)
        st.rerun()

    if st.session_state.judge_score:
        st.subheader("Attending Board Review")
        st.markdown(
            f"""
- **Score:** {st.session_state.judge_score.score}/5
- **Label:** {st.session_state.judge_score.label}
- **Reasoning:** {st.session_state.judge_score.reasoning}
"""
        )
        st.subheader("Reference Diagnosis")
        st.markdown(f"**{case.ground_truth_diagnosis}**")
        st.subheader("Encounter Transcript")
        transcript_text = build_transcript(case)
        st.code(transcript_text, language="markdown")

    st.subheader("Download Complete Transcript")
    transcript_text = build_transcript(case)
    st.download_button(
        label="Download Clinical Transcript",
        data=transcript_text,
        file_name=f"{case.case_id}_MultiTurnDiagnosisCopilot.txt",
        mime="text/plain",
    )


if __name__ == "__main__":
    main()


