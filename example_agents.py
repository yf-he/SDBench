"""Example diagnostic agents for SDBench testing."""

import random
from typing import List
from data_models import AgentAction, ActionType
from sdbench import DiagnosticAgent
from config import Config
from utils.llm_client import chat_completion_with_retries

class RandomDiagnosticAgent(DiagnosticAgent):
    """A random diagnostic agent for baseline testing."""
    
    def __init__(self, name: str = "RandomAgent"):
        super().__init__(name)
        self.actions_taken = 0
        self.max_actions = 10
    
    def get_next_action(self, case_abstract: str, encounter_history: List[AgentAction]) -> AgentAction:
        """Generate a random action."""
        self.actions_taken += 1
        
        # If we've taken too many actions, make a random diagnosis
        if self.actions_taken >= self.max_actions:
            return AgentAction(
                action_type=ActionType.DIAGNOSE,
                content="I cannot determine the diagnosis with the available information."
            )
        
        # Randomly choose action type
        action_type = random.choice([ActionType.ASK_QUESTIONS, ActionType.REQUEST_TESTS])
        
        if action_type == ActionType.ASK_QUESTIONS:
            questions = [
                "What is the patient's age and gender?",
                "What are the main symptoms?",
                "How long have the symptoms been present?",
                "Are there any associated symptoms?",
                "What is the patient's medical history?",
                "Are there any recent exposures or travel?",
                "What medications is the patient taking?",
                "Are there any allergies?",
                "What are the vital signs?",
                "Are there any physical examination findings?"
            ]
            content = random.choice(questions)
        
        else:  # REQUEST_TESTS
            tests = [
                "Complete Blood Count",
                "Comprehensive Metabolic Panel",
                "Chest X-ray",
                "CT scan of the chest",
                "Blood cultures",
                "Urinalysis",
                "Electrocardiogram",
                "Echocardiogram",
                "Liver function tests",
                "Thyroid function tests"
            ]
            content = random.choice(tests)
        
        return AgentAction(action_type=action_type, content=content)
    
    def reset(self) -> None:
        """Reset for new case."""
        self.actions_taken = 0

class LLMDiagnosticAgent(DiagnosticAgent):
    """A diagnostic agent powered by a language model."""
    
    def __init__(self, name: str = "LLMAgent", config: Config = None):
        super().__init__(name)
        self.config = config or Config()
        self.client = self.config.get_openai_client()
        self.model = self.config.GATEKEEPER_MODEL
        self.actions_taken = 0
        self.max_actions = 20
        self.diagnostic_hypotheses = []
    
    def get_next_action(self, case_abstract: str, encounter_history: List[AgentAction]) -> AgentAction:
        """Generate an intelligent action using LLM."""
        self.actions_taken += 1
        
        # If approaching the cap, force a final diagnosis next
        if self.actions_taken >= self.max_actions - 1:
            return self._make_final_diagnosis(case_abstract, encounter_history)
        
        # Build context from encounter history
        context = self._build_context(case_abstract, encounter_history)
        
        # Generate next action
        action = self._generate_next_action(context)
        # If still not diagnosing and near cap, finalize
        if action.action_type != ActionType.DIAGNOSE and self.actions_taken >= self.max_actions - 1:
            return self._make_final_diagnosis(case_abstract, encounter_history)
        return action
    
    def _build_context(self, case_abstract: str, encounter_history: List[AgentAction]) -> str:
        """Build context string from case and history."""
        context = f"Case Abstract: {case_abstract}\n\n"
        
        if encounter_history:
            context += "Encounter History:\n"
            for i, action in enumerate(encounter_history, 1):
                context += f"{i}. {action.action_type.value}: {action.content}\n"
        
        return context
    
    def _generate_next_action(self, context: str) -> AgentAction:
        """Generate the next action using LLM."""
        prompt = f"""
        You are an expert diagnostic physician. Based on the case information and encounter history, determine the next most appropriate action.

        {context}

        Iterative policy (soft guidance):
        - Prefer asking a few (e.g., 3–8) targeted, high-yield questions/tests before committing.
        - Prioritize clarifying red flags, key differentials, and decisive tests.
        - When you are reasonably confident, finalize succinctly.

        If the case provides OPTIONS (A-D), prefer selecting a final answer from those options when you are confident.

        You can take one of three types of actions:
        1. Ask a specific question about the patient's history or examination
        2. Order a specific test or procedure
        3. Make a final diagnosis (only if you're confident). If options A-D are present, pick ONE option as final.

        Respond in one of the following formats:
        - For questions: <question>Your specific question here</question>
        - For tests: <test>Specific test name here</test>
        - For diagnosis WITHOUT options: <diagnosis>Your diagnosis here</diagnosis>
        - For diagnosis WITH options: <diagnosis_option>A</diagnosis_option> <diagnosis>The full text of the chosen option</diagnosis>

        Be specific and clinical in your requests. Avoid vague or overly broad questions/tests.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            
            action_text = response.choices[0].message.content.strip()
            return self._parse_action_text(action_text)
            
        except Exception as e:
            print(f"Error generating action: {e}")
            # Fallback to random action
            return self._fallback_action()
    
    def _parse_action_text(self, action_text: str) -> AgentAction:
        """Parse action text into AgentAction object."""
        import re
        
        # Look for diagnosis option + text
        opt_match = re.search(r'<diagnosis_option>\s*([ABCD])\s*</diagnosis_option>', action_text, re.IGNORECASE)
        diag_match = re.search(r'<diagnosis>(.*?)</diagnosis>', action_text, re.DOTALL)
        if opt_match and diag_match:
            choice = opt_match.group(1).upper()
            diagnosis_text = diag_match.group(1).strip()
            return AgentAction(
                action_type=ActionType.DIAGNOSE,
                content=f"Option {choice}: {diagnosis_text}"
            )

        # Look for question tags
        question_match = re.search(r'<question>(.*?)</question>', action_text, re.DOTALL)
        if question_match:
            return AgentAction(
                action_type=ActionType.ASK_QUESTIONS,
                content=question_match.group(1).strip()
            )
        
        # Look for test tags
        test_match = re.search(r'<test>(.*?)</test>', action_text, re.DOTALL)
        if test_match:
            return AgentAction(
                action_type=ActionType.REQUEST_TESTS,
                content=test_match.group(1).strip()
            )
        
        # Look for diagnosis tags
        diagnosis_match = re.search(r'<diagnosis>(.*?)</diagnosis>', action_text, re.DOTALL)
        if diagnosis_match:
            return AgentAction(
                action_type=ActionType.DIAGNOSE,
                content=diagnosis_match.group(1).strip()
            )
        
        # If no tags found, treat as question
        return AgentAction(
            action_type=ActionType.ASK_QUESTIONS,
            content=action_text.strip()
        )
    
    def _make_final_diagnosis(self, case_abstract: str, encounter_history: List[AgentAction]) -> AgentAction:
        """Make a final diagnosis when max actions reached."""
        context = self._build_context(case_abstract, encounter_history)
        
        prompt = f"""
        You are an expert diagnostic physician. Based on all available information, make your best diagnostic assessment.

        {context}

        If OPTIONS (A-D) are provided in the case context, you MUST choose exactly one option as the final answer.

        Respond in one of the following formats:
        - If options present: <diagnosis_option>[A-D]</diagnosis_option> <diagnosis>The full text of the chosen option</diagnosis>
        - If no options present: <diagnosis>Your diagnosis here</diagnosis>
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.2
            )
            
            action_text = response.choices[0].message.content.strip()
            return self._parse_action_text(action_text)
            
        except Exception as e:
            print(f"Error making final diagnosis: {e}")
            return AgentAction(
                action_type=ActionType.DIAGNOSE,
                content="Unable to determine diagnosis with available information."
            )
    
    def _fallback_action(self) -> AgentAction:
        """Fallback action when LLM fails."""
        fallback_questions = [
            "What are the patient's vital signs?",
            "What is the patient's medical history?",
            "Are there any physical examination findings?",
            "What medications is the patient taking?"
        ]
        
        return AgentAction(
            action_type=ActionType.ASK_QUESTIONS,
            content=random.choice(fallback_questions)
        )
    
    def reset(self) -> None:
        """Reset for new case."""
        self.actions_taken = 0
        self.diagnostic_hypotheses = []

class ConservativeDiagnosticAgent(DiagnosticAgent):
    """A conservative diagnostic agent that asks many questions before testing."""
    
    def __init__(self, name: str = "ConservativeAgent"):
        super().__init__(name)
        self.actions_taken = 0
        self.questions_asked = 0
        self.tests_ordered = 0
        self.max_questions = 8
        self.max_tests = 5
        self.max_total_actions = 15
    
    def get_next_action(self, case_abstract: str, encounter_history: List[AgentAction]) -> AgentAction:
        """Generate a conservative action."""
        self.actions_taken += 1
        
        # If we've taken too many actions, make a diagnosis
        if self.actions_taken >= self.max_total_actions:
            return AgentAction(
                action_type=ActionType.DIAGNOSE,
                content="Based on available information, I cannot make a definitive diagnosis."
            )
        
        # Prefer questions over tests
        if self.questions_asked < self.max_questions:
            return self._ask_question()
        elif self.tests_ordered < self.max_tests:
            return self._order_test()
        else:
            return AgentAction(
                action_type=ActionType.DIAGNOSE,
                content="Based on available information, I cannot make a definitive diagnosis."
            )
    
    def _ask_question(self) -> AgentAction:
        """Ask a conservative question."""
        self.questions_asked += 1
        
        questions = [
            "What is the patient's detailed medical history?",
            "What are all the symptoms and their progression?",
            "What medications and allergies does the patient have?",
            "What are the complete vital signs?",
            "What are all the physical examination findings?",
            "Are there any recent exposures or travel?",
            "What is the patient's family history?",
            "Are there any associated symptoms or triggers?"
        ]
        
        question_index = min(self.questions_asked - 1, len(questions) - 1)
        content = questions[question_index]
        
        return AgentAction(action_type=ActionType.ASK_QUESTIONS, content=content)
    
    def _order_test(self) -> AgentAction:
        """Order a conservative test."""
        self.tests_ordered += 1
        
        tests = [
            "Complete Blood Count with differential",
            "Comprehensive Metabolic Panel",
            "Chest X-ray",
            "Basic metabolic panel",
            "Urinalysis"
        ]
        
        test_index = min(self.tests_ordered - 1, len(tests) - 1)
        content = tests[test_index]
        
        return AgentAction(action_type=ActionType.REQUEST_TESTS, content=content)
    
    def reset(self) -> None:
        """Reset for new case."""
        self.actions_taken = 0
        self.questions_asked = 0
        self.tests_ordered = 0

class AggressiveDiagnosticAgent(DiagnosticAgent):
    """An aggressive diagnostic agent that orders many tests quickly."""
    
    def __init__(self, name: str = "AggressiveAgent"):
        super().__init__(name)
        self.actions_taken = 0
        self.tests_ordered = 0
        self.max_tests = 10
        self.max_total_actions = 12
    
    def get_next_action(self, case_abstract: str, encounter_history: List[AgentAction]) -> AgentAction:
        """Generate an aggressive action."""
        self.actions_taken += 1
        
        # If we've taken too many actions, make a diagnosis
        if self.actions_taken >= self.max_total_actions:
            return AgentAction(
                action_type=ActionType.DIAGNOSE,
                content="Based on available information, I cannot make a definitive diagnosis."
            )
        
        # Prefer tests over questions
        if self.tests_ordered < self.max_tests:
            return self._order_test()
        else:
            return AgentAction(
                action_type=ActionType.DIAGNOSE,
                content="Based on available information, I cannot make a definitive diagnosis."
            )
    
    def _order_test(self) -> AgentAction:
        """Order an aggressive test."""
        self.tests_ordered += 1
        
        tests = [
            "Complete Blood Count with differential",
            "Comprehensive Metabolic Panel",
            "Chest X-ray",
            "CT scan of the chest with contrast",
            "Blood cultures",
            "Urinalysis",
            "Electrocardiogram",
            "Echocardiogram",
            "Liver function tests",
            "Thyroid function tests",
            "CT scan of the abdomen with contrast",
            "MRI of the brain"
        ]
        
        test_index = min(self.tests_ordered - 1, len(tests) - 1)
        content = tests[test_index]
        
        return AgentAction(action_type=ActionType.REQUEST_TESTS, content=content)
    
    def reset(self) -> None:
        """Reset for new case."""
        self.actions_taken = 0
        self.tests_ordered = 0


class MultiLLMDxOAgent(DiagnosticAgent):
    """Five-LLM MAI-DxO: each role is handled by its own (potentially different) model.

    Roles -> models (default to gatekeeper model if not provided):
      hypothesis_model, test_chooser_model, challenger_model, stewardship_model, checklist_model
    """

    def __init__(self,
                 name: str = "MAI-DxO(5xLLM)",
                 config: Config = None,
                 # Convenience: set one model for all roles
                 model_for_all: str = None,
                 # Or customize per-role (overrides model_for_all if provided)
                 hypothesis_model: str = None,
                 test_chooser_model: str = None,
                 challenger_model: str = None,
                 stewardship_model: str = None,
                 checklist_model: str = None):
        super().__init__(name)
        self.config = config or Config()
        self.client = self.config.get_openai_client()
        base = self.config.GATEKEEPER_MODEL
        if model_for_all:
            self.models = {
                'hypothesis': model_for_all,
                'test_chooser': model_for_all,
                'challenger': model_for_all,
                'stewardship': model_for_all,
                'checklist': model_for_all,
            }
        else:
            self.models = {
                'hypothesis': hypothesis_model or base,
                'test_chooser': test_chooser_model or base,
                'challenger': challenger_model or base,
                'stewardship': stewardship_model or base,
                'checklist': checklist_model or base,
            }
        self.actions_taken = 0
        self.max_actions = 20
        self.panel_trace = []
        self.debate_rounds = 0
        self.min_debate_rounds = 2  # require at least 2 debate rounds before allowing diagnosis

    def reset(self) -> None:
        self.actions_taken = 0
        self.panel_trace = []
        self.debate_rounds = 0

    def get_next_action(self, case_abstract: str, encounter_history: List[AgentAction]) -> AgentAction:
        self.actions_taken += 1
        self.debate_rounds += 1
        context = self._build_context(case_abstract, encounter_history)
        # 1) Hypothesis
        hyp = self._call_role('hypothesis', f"""
You are Dr. Hypothesis. Maintain a probability-ranked top-3 differential diagnosis and briefly justify.

{context}

Round: {self.debate_rounds}
Rule: Do NOT finalize diagnosis in round 1; focus on building a good differential.

Return format:
1) DxA: pA
2) DxB: pB
3) DxC: pC
Notes: one sentence rationale.
""")
        # 2) Test-Chooser
        tc = self._call_role('test_chooser', f"""
You are Dr. Test-Chooser. Given the case and hypotheses below, propose up to three tests that maximally discriminate.

{context}

Hypotheses:
{hyp}

Round: {self.debate_rounds}
Rule: On round 1, prefer questions or low-cost, high-yield tests; do NOT finalize diagnosis.

Return format as bullet list (<=3):
- Test 1
- Test 2
- Test 3
""")
        # 3) Challenger
        ch = self._call_role('challenger', f"""
You are Dr. Challenger. Identify potential anchoring bias or contradictions; propose one falsifying test if applicable.

{context}

Hypotheses:
{hyp}
Proposed tests:
{tc}

Return 2-4 bullet points.
""")
        # 4) Stewardship
        st = self._call_role('stewardship', f"""
You are Dr. Stewardship. Ensure cost-conscious care; suggest cheaper equivalents for proposed tests and veto low-yield options.

{context}
Tests proposed:
{tc}
Challenger notes:
{ch}

Return a final approved test list (<=3), one per line. If equally good cheaper alternatives exist, use the cheaper names.
""")
        # 5) Checklist
        ck = self._call_role('checklist', f"""
You are Dr. Checklist. Validate test names are specific and billable; ensure internal consistency. If options (A-D) exist in the abstract/context and confidence is high, you may suggest a single choice.

{context}
Final approved tests:
{st}

Round: {self.debate_rounds}
Constraint: Do NOT choose the diagnose action in round 1. Choose question or test.

Return EXACTLY this block (missing fields empty if not used):
<check>
  <approved_tests>Test 1; Test 2; Test 3</approved_tests>
  <decision>question|test|diagnose</decision>
  <question>...</question>
  <diagnosis_option>A</diagnosis_option>
  <diagnosis>Text</diagnosis>
</check>
""")
        self.panel_trace = [hyp, tc, ch, st, ck]
        act = self._parse_check_block(ck)
        # Enforce minimum debate rounds before diagnosing
        if act.action_type == ActionType.DIAGNOSE and self.debate_rounds < self.min_debate_rounds:
            # Convert to a clarifying question sourced from challenger/checklist signal
            followup_q = "Please clarify key red flags and timeline; provide one specific question."
            return AgentAction(action_type=ActionType.ASK_QUESTIONS, content=followup_q)
        return act

    def _build_context(self, case_abstract: str, encounter_history: List[AgentAction]) -> str:
        ctx = f"Case Abstract: {case_abstract}\n\n"
        if encounter_history:
            ctx += "Encounter History:\n"
            for i, action in enumerate(encounter_history, 1):
                ctx += f"{i}. {action.action_type.value}: {action.content}\n"
        return ctx

    def _call_role(self, role_key: str, content: str) -> str:
        try:
            response = chat_completion_with_retries(
                client=self.client,
                model=self.models[role_key],
                messages=[{"role": "user", "content": content}],
                max_retries=4,
                retry_interval_sec=6,
                max_tokens=500,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"{role_key} role error: {e}")
            return ""

    def _parse_check_block(self, text: str) -> AgentAction:
        import re
        decision = re.search(r"<decision>\s*(question|test|diagnose)\s*</decision>", text, re.IGNORECASE)
        decision = decision.group(1).lower() if decision else "question"

        if decision == "question":
            q = re.search(r"<question>(.*?)</question>", text, re.DOTALL)
            content = q.group(1).strip() if q else "Could you clarify your key symptoms and timeline?"
            return AgentAction(action_type=ActionType.ASK_QUESTIONS, content=content)

        if decision == "test":
            t = re.search(r"<approved_tests>(.*?)</approved_tests>", text, re.DOTALL)
            tests = (t.group(1).strip() if t else "")
            first = tests.split(";")[0].strip() if tests else "Complete Blood Count with differential"
            return AgentAction(action_type=ActionType.REQUEST_TESTS, content=first)

        if decision == "diagnose":
            opt = re.search(r"<diagnosis_option>\s*([ABCD])\s*</diagnosis_option>", text, re.IGNORECASE)
            diag = re.search(r"<diagnosis>(.*?)</diagnosis>", text, re.DOTALL)
            if opt and diag:
                return AgentAction(action_type=ActionType.DIAGNOSE, content=f"Option {opt.group(1).upper()}: {diag.group(1).strip()}")
            if diag:
                return AgentAction(action_type=ActionType.DIAGNOSE, content=diag.group(1).strip())
            return AgentAction(action_type=ActionType.DIAGNOSE, content="Unable to determine.")

        return AgentAction(action_type=ActionType.ASK_QUESTIONS, content="Please provide more history about onset, progression, and associated symptoms.")


class MAIDxOAgent(DiagnosticAgent):
    """Multi-Agent Iterative Diagnosis Orchestrator (MAI-DxO).

    Simulates a panel of five personas:
      - Dr. Hypothesis
      - Dr. Test-Chooser
      - Dr. Challenger
      - Dr. Stewardship
      - Dr. Checklist

    Produces one of: ask question, order tests (<=3), or diagnose (option A-D or free-text).
    """

    def __init__(self, name: str = "MAI-DxO", config: Config = None):
        super().__init__(name)
        self.config = config or Config()
        self.client = self.config.get_openai_client()
        self.model = self.config.GATEKEEPER_MODEL
        self.actions_taken = 0
        self.max_actions = 20
        self.panel_memory = ""
        self.panel_rounds = []
        self.running_cost_estimate = 0.0

    def get_next_action(self, case_abstract: str, encounter_history: List[AgentAction]) -> AgentAction:
        self.actions_taken += 1
        context = self._build_context(case_abstract, encounter_history)
        force_diag = (self.actions_taken >= self.max_actions - 1)
        proposal = self._panel_deliberation(context, force_diagnose=force_diag)
        try:
            self.panel_rounds.append(proposal)
        except Exception:
            pass
        act = self._parse_panel_output(proposal)
        if act.action_type != ActionType.DIAGNOSE and self.actions_taken >= self.max_actions - 1:
            return self._force_final_diagnosis_maidxo(context)
        return act

    def reset(self) -> None:
        self.actions_taken = 0
        self.panel_memory = ""
        self.panel_rounds = []
        self.running_cost_estimate = 0.0

    def _build_context(self, case_abstract: str, encounter_history: List[AgentAction]) -> str:
        context = f"Case Abstract: {case_abstract}\n\n"
        if encounter_history:
            context += "Encounter History:\n"
            for i, action in enumerate(encounter_history, 1):
                context += f"{i}. {action.action_type.value}: {action.content}\n"
        if self.panel_memory:
            context += f"\nPanel Notes (memory):\n{self.panel_memory}\n"
        return context

    def _panel_deliberation(self, context: str, force_diagnose: bool = False) -> str:
        prompt = f"""
You are a virtual panel of five doctors collaborating on diagnosis. Follow roles and emit a structured plan.

{context}

Panel roles:
- Dr. Hypothesis: Maintain top-3 differential with probabilities (Bayesian update after new info).
- Dr. Test-Chooser: Propose up to 3 tests that maximally discriminate between leading hypotheses.
- Dr. Challenger: Point out anchoring bias, contradictory evidence, and propose tests to falsify.
- Dr. Stewardship: Ensure cost-consciousness; suggest cheaper but equivalent alternatives; veto low-yield expensive tests.
- Dr. Checklist: Validate test names are specific and billable; enforce output schema; ensure internal consistency.

Iterative policy (soft guidance):
- Prefer asking a few (e.g., 3–8) targeted, high-yield questions/tests before committing.
- Prioritize clarifying red flags, key differentials, and decisive tests.
- When reasonably confident, finalize succinctly.

Chain of Debate (keep concise):
1) Hypothesis update (top3 with probabilities summing to 1.0)
2) Test-Chooser proposal (<=3 tests)
3) Challenger critique
4) Stewardship adjustments (cost-aware)
5) Checklist validation

Decision rule:
- If reasonably confident AND options (A-D) exist, pick ONE option.
- Else if not confident, choose to ask an informative question OR order <=3 decisive tests.
{'- On this round you MUST finalize with a diagnose action (choose ONE option if present).' if force_diagnose else ''}

Return EXACTLY this schema (no extra text):
<panel>
  <hypotheses>
    1) DxA: pA
    2) DxB: pB
    3) DxC: pC
  </hypotheses>
  <tests>
    - Test 1
    - Test 2
    - Test 3
  </tests>
  <decision>
    <action>question|test|diagnose</action>
    <question>...</question>
    <ordered_tests>Test 1; Test 2</ordered_tests>
    <diagnosis_option>A</diagnosis_option>
    <diagnosis>Full text of the chosen option or final dx</diagnosis>
  </decision>
  <notes>1-2 line rationale</notes>
</panel>
"""
        try:
            response = chat_completion_with_retries(
                client=self.client,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_retries=5,
                retry_interval_sec=8,
                max_tokens=600,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"MAI-DxO deliberation error: {e}")
            return ""

    def _parse_panel_output(self, text: str) -> AgentAction:
        import re
        mnotes = re.search(r"<notes>(.*?)</notes>", text, re.DOTALL)
        if mnotes:
            note = mnotes.group(1).strip()
            self.panel_memory = (self.panel_memory + "\n" + note).strip() if self.panel_memory else note

        action_match = re.search(r"<action>\s*(question|test|diagnose)\s*</action>", text, re.IGNORECASE)
        action = action_match.group(1).lower() if action_match else "question"

        if action == "question":
            q = re.search(r"<question>(.*?)</question>", text, re.DOTALL)
            content = (q.group(1).strip() if q else "Could you clarify your key symptoms and their timeline?")
            return AgentAction(action_type=ActionType.ASK_QUESTIONS, content=content)

        if action == "test":
            t = re.search(r"<ordered_tests>(.*?)</ordered_tests>", text, re.DOTALL)
            tests = (t.group(1).strip() if t else "")
            first = tests.split(";")[0].strip() if tests else "Complete Blood Count with differential"
            return AgentAction(action_type=ActionType.REQUEST_TESTS, content=first)

        if action == "diagnose":
            opt = re.search(r"<diagnosis_option>\s*([ABCD])\s*</diagnosis_option>", text, re.IGNORECASE)
            diag = re.search(r"<diagnosis>(.*?)</diagnosis>", text, re.DOTALL)
            if opt and diag:
                return AgentAction(action_type=ActionType.DIAGNOSE, content=f"Option {opt.group(1).upper()}: {diag.group(1).strip()}")
            if diag:
                return AgentAction(action_type=ActionType.DIAGNOSE, content=diag.group(1).strip())
            return AgentAction(action_type=ActionType.DIAGNOSE, content="Unable to determine.")

        return AgentAction(action_type=ActionType.ASK_QUESTIONS, content="Can you provide further details on onset, progression, and associated symptoms?")

    def _force_final_diagnosis_maidxo(self, context: str) -> AgentAction:
        """Force the panel to output a mandatory final diagnosis (choose option if present)."""
        content = f"""
You MUST finalize the diagnosis now. If OPTIONS (A-D) are present, choose exactly ONE as the final answer.

{context}

Return EXACTLY this block:
<panel>
  <decision>
    <action>diagnose</action>
    <diagnosis_option>A</diagnosis_option>
    <diagnosis>Full text of the chosen option or final dx</diagnosis>
  </decision>
</panel>
"""
        try:
            response = chat_completion_with_retries(
                client=self.client,
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_retries=4,
                retry_interval_sec=6,
                max_tokens=300,
                temperature=0.2,
            )
            text = response.choices[0].message.content.strip()
            # Reuse existing parse for diagnose branch
            import re
            diag = re.search(r"<diagnosis>(.*?)</diagnosis>", text, re.DOTALL)
            opt = re.search(r"<diagnosis_option>\s*([ABCD])\s*</diagnosis_option>", text, re.IGNORECASE)
            if opt and diag:
                return AgentAction(action_type=ActionType.DIAGNOSE, content=f"Option {opt.group(1).upper()}: {diag.group(1).strip()}")
            if diag:
                return AgentAction(action_type=ActionType.DIAGNOSE, content=diag.group(1).strip())
            return AgentAction(action_type=ActionType.DIAGNOSE, content="Unable to determine.")
        except Exception:
            return AgentAction(action_type=ActionType.DIAGNOSE, content="Unable to determine.")
