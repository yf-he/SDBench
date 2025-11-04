"""Gatekeeper Agent implementation for SDBench."""

import re
from typing import List, Optional, Tuple
from data_models import CaseFile, AgentAction, GatekeeperResponse, ActionType
from config import Config
from utils.llm_client import chat_completion_with_retries

class GatekeeperAgent:
    """The Gatekeeper Agent serves as the information oracle for patient cases."""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = config.get_openai_client()
        self.model = config.GATEKEEPER_MODEL
    
    def process_action(self, action: AgentAction, case_file: CaseFile) -> GatekeeperResponse:
        """Process an action from the diagnostic agent and return a response."""
        if action.action_type == ActionType.ASK_QUESTIONS:
            return self._handle_question(action.content, case_file)
        elif action.action_type == ActionType.REQUEST_TESTS:
            return self._handle_test_request(action.content, case_file)
        else:
            raise ValueError(f"Gatekeeper cannot process action type: {action.action_type}")
    
    def _handle_question(self, question: str, case_file: CaseFile) -> GatekeeperResponse:
        """Handle a question from the diagnostic agent."""
        # Check if the answer is explicitly in the case file
        answer = self._extract_explicit_answer(question, case_file.full_case_text)
        
        if answer:
            return GatekeeperResponse(
                response_text=answer,
                is_synthetic=False
            )
        else:
            # Generate synthetic answer
            synthetic_answer = self._generate_synthetic_answer(question, case_file)
            return GatekeeperResponse(
                response_text=synthetic_answer,
                is_synthetic=True
            )
    
    def _handle_test_request(self, test_request: str, case_file: CaseFile) -> GatekeeperResponse:
        """Handle a test request from the diagnostic agent."""
        # Check if the test result is explicitly in the case file
        result = self._extract_explicit_test_result(test_request, case_file.full_case_text)
        
        if result:
            return GatekeeperResponse(
                response_text=result,
                is_synthetic=False
            )
        else:
            # Generate synthetic test result
            synthetic_result = self._generate_synthetic_test_result(test_request, case_file)
            return GatekeeperResponse(
                response_text=synthetic_result,
                is_synthetic=True
            )
    
    def _extract_explicit_answer(self, question: str, case_text: str) -> Optional[str]:
        """Extract explicit answer from case text if available."""
        # Use LLM to determine if the answer is explicitly available
        prompt = f"""
        You are a medical information extractor. Given a clinical case text and a specific question, determine if the answer is explicitly stated in the case text.
        
        Question: {question}
        
        Case Text: {case_text[:2000]}...
        
        If the answer is explicitly stated, provide ONLY the relevant excerpt from the case text. If not explicitly stated, respond with "NOT_EXPLICIT".
        
        Response:
        """
        
        try:
            response = chat_completion_with_retries(
                client=self.client,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_retries=5,
                retry_interval_sec=8,
                max_tokens=500,
                temperature=0.1,
            )
            
            answer = response.choices[0].message.content.strip()
            if answer == "NOT_EXPLICIT":
                return None
            return answer
        except Exception as e:
            print("Error extracting explicit answer:")
            print(f"  ErrorType: {type(e).__name__}")
            print(f"  ErrorRepr: {e!r}")
            return None
    
    def _extract_explicit_test_result(self, test_request: str, case_text: str) -> Optional[str]:
        """Extract explicit test result from case text if available."""
        # Use LLM to determine if the test result is explicitly available
        prompt = f"""
        You are a medical information extractor. Given a clinical case text and a specific test request, determine if the test result is explicitly stated in the case text.
        
        Test Request: {test_request}
        
        Case Text: {case_text[:2000]}...
        
        If the test result is explicitly stated, provide ONLY the relevant excerpt from the case text. If not explicitly stated, respond with "NOT_EXPLICIT".
        
        Response:
        """
        
        try:
            response = chat_completion_with_retries(
                client=self.client,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_retries=5,
                retry_interval_sec=8,
                max_tokens=500,
                temperature=0.1,
            )
            
            result = response.choices[0].message.content.strip()
            if result == "NOT_EXPLICIT":
                return None
            return result
        except Exception as e:
            print("Error extracting explicit test result:")
            print(f"  ErrorType: {type(e).__name__}")
            print(f"  ErrorRepr: {e!r}")
            return None
    
    def _generate_synthetic_answer(self, question: str, case_file: CaseFile) -> str:
        """Generate a synthetic answer that's consistent with the case."""
        prompt = f"""
        You are a medical information oracle. Generate a plausible, synthetic answer to a clinical question that is consistent with the patient's overall clinical picture and the final diagnosis.
        
        Question: {question}
        
        Case Context:
        Initial Abstract: {case_file.initial_abstract}
        
        Full Case Text: {case_file.full_case_text[:3000]}...
        
        Final Diagnosis: {case_file.ground_truth_diagnosis}
        
        Generate a realistic, objective clinical finding that would be consistent with this patient's presentation and final diagnosis. Do not provide diagnostic interpretations or hints. Only provide the objective finding as if it were a real clinical observation.
        
        Response:
        """
        
        try:
            response = chat_completion_with_retries(
                client=self.client,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_retries=5,
                retry_interval_sec=8,
                max_tokens=300,
                temperature=0.7,
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print("Error generating synthetic answer:")
            print(f"  ErrorType: {type(e).__name__}")
            print(f"  ErrorRepr: {e!r}")
            return "Unable to provide information at this time."
    
    def _generate_synthetic_test_result(self, test_request: str, case_file: CaseFile) -> str:
        """Generate a synthetic test result that's consistent with the case."""
        prompt = f"""
        You are a medical information oracle. Generate a plausible, synthetic test result that is consistent with the patient's overall clinical picture and the final diagnosis.
        
        Test Request: {test_request}
        
        Case Context:
        Initial Abstract: {case_file.initial_abstract}
        
        Full Case Text: {case_file.full_case_text[:3000]}...
        
        Final Diagnosis: {case_file.ground_truth_diagnosis}
        
        Generate a realistic, objective test result that would be consistent with this patient's presentation and final diagnosis. Format it as a typical clinical report. Do not provide diagnostic interpretations or hints. Only provide the objective findings as if they were real test results.
        
        Response:
        """
        
        try:
            response = chat_completion_with_retries(
                client=self.client,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_retries=5,
                retry_interval_sec=8,
                max_tokens=400,
                temperature=0.7,
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print("Error generating synthetic test result:")
            print(f"  ErrorType: {type(e).__name__}")
            print(f"  ErrorRepr: {e!r}")
            return "Test result not available at this time."
    
    def validate_request(self, action: AgentAction) -> Tuple[bool, str]:
        """Validate if a request is appropriate for the gatekeeper."""
        if action.action_type == ActionType.ASK_QUESTIONS:
            # Check for overly broad questions
            broad_indicators = [
                "tell me everything", "what's wrong", "what should I do",
                "give me all information", "summarize the case"
            ]
            
            question_lower = action.content.lower()
            for indicator in broad_indicators:
                if indicator in question_lower:
                    return False, "Please ask more specific questions about the patient's history or examination findings."
            
            return True, ""
        
        elif action.action_type == ActionType.REQUEST_TESTS:
            # Check for vague test requests
            vague_indicators = [
                "run blood work", "do some imaging", "order labs",
                "get tests", "run diagnostics"
            ]
            
            test_lower = action.content.lower()
            for indicator in vague_indicators:
                if indicator in test_lower:
                    return False, "Please specify the exact test you would like to order (e.g., 'Complete Blood Count', 'CT of the abdomen with contrast')."
            
            return True, ""
        
        return True, ""
