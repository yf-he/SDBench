"""Example diagnostic agents for SDBench testing."""

import random
from typing import List
from openai import OpenAI
from data_models import AgentAction, ActionType
from sdbench import DiagnosticAgent
from config import Config

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
        self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)
        self.model = self.config.GATEKEEPER_MODEL
        self.actions_taken = 0
        self.max_actions = 15
        self.diagnostic_hypotheses = []
    
    def get_next_action(self, case_abstract: str, encounter_history: List[AgentAction]) -> AgentAction:
        """Generate an intelligent action using LLM."""
        self.actions_taken += 1
        
        # If we've taken too many actions, make a diagnosis
        if self.actions_taken >= self.max_actions:
            return self._make_final_diagnosis(case_abstract, encounter_history)
        
        # Build context from encounter history
        context = self._build_context(case_abstract, encounter_history)
        
        # Generate next action
        action = self._generate_next_action(context)
        
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

        You can take one of three types of actions:
        1. Ask a specific question about the patient's history or examination
        2. Order a specific test or procedure
        3. Make a final diagnosis (only if you're confident)

        Choose the most appropriate action and respond in the following format:
        - For questions: <question>Your specific question here</question>
        - For tests: <test>Specific test name here</test>
        - For diagnosis: <diagnosis>Your diagnosis here</diagnosis>

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

        Provide your final diagnosis in the format: <diagnosis>Your diagnosis here</diagnosis>

        Be specific and include the most likely diagnosis based on the available information.
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
