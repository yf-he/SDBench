"""Judge Agent implementation for SDBench."""

from typing import List
from openai import OpenAI
from data_models import CaseFile, JudgeScore
from config import Config

class JudgeAgent:
    """The Judge Agent evaluates final diagnoses using a 5-point Likert scale."""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.JUDGE_MODEL
    
    def evaluate_diagnosis(self, candidate_diagnosis: str, case_file: CaseFile) -> JudgeScore:
        """Evaluate a candidate diagnosis against the ground truth."""
        prompt = self._create_evaluation_prompt(candidate_diagnosis, case_file)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )
            
            result = self._parse_evaluation_response(response.choices[0].message.content.strip())
            return result
            
        except Exception as e:
            print(f"Error evaluating diagnosis: {e}")
            return JudgeScore(
                score=1,
                reasoning="Error in evaluation process",
                label="Completely incorrect"
            )
    
    def _create_evaluation_prompt(self, candidate_diagnosis: str, case_file: CaseFile) -> str:
        """Create the evaluation prompt for the judge agent."""
        return f"""
        You are an expert medical judge evaluating diagnostic accuracy. You must score the candidate diagnosis on a 5-point Likert scale using the exact rubric provided below.

        CANDIDATE DIAGNOSIS: {candidate_diagnosis}

        GROUND TRUTH DIAGNOSIS: {case_file.ground_truth_diagnosis}

        CASE CONTEXT:
        Initial Abstract: {case_file.initial_abstract}

        Full Case Details: {case_file.full_case_text[:4000]}...

        EVALUATION RUBRIC:
        Score 5 - Perfect/Clinically superior: Clinically identical to the reference or a strictly more specific version. Any added detail must be directly related. No unrelated or incorrect additions.

        Score 4 - Mostly correct (minor incompleteness): Core disease correctly identified but a secondary qualifier is missing or slightly mis-specified. Overall management would remain largely unchanged.

        Score 3 - Partially correct (major error): Correct general disease category, but a major error in etiology, site, or critical specificity. Would alter work-up or prognosis.

        Score 2 - Largely incorrect: Shares superficial features only (e.g., manifestation without etiology). Fundamentally misdirects clinical work-up or partially contradicts case details.

        Score 1 - Completely incorrect: No meaningful overlap; wrong organ/system; nonsensical or contradictory. Following this diagnosis would likely lead to harmful care.

        INSTRUCTIONS:
        1. Carefully compare the candidate diagnosis with the ground truth diagnosis
        2. Consider the clinical context and case details
        3. Assign a score from 1-5 based on the rubric above
        4. Provide clear reasoning for your score
        5. Identify the appropriate label for the score

        Respond in the following JSON format:
        {{
            "score": [1-5],
            "reasoning": "Detailed explanation of your evaluation",
            "label": "Exact label from the rubric"
        }}
        """
    
    def _parse_evaluation_response(self, response: str) -> JudgeScore:
        """Parse the evaluation response from the judge agent."""
        import json
        import re
        
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                return JudgeScore(
                    score=int(result.get("score", 1)),
                    reasoning=result.get("reasoning", "No reasoning provided"),
                    label=result.get("label", "Completely incorrect")
                )
        except Exception as e:
            print(f"Error parsing JSON response: {e}")
        
        # Fallback parsing if JSON extraction fails
        try:
            # Look for score in the text
            score_match = re.search(r'score["\s]*:[\s]*(\d+)', response, re.IGNORECASE)
            score = int(score_match.group(1)) if score_match else 1
            
            # Look for reasoning
            reasoning_match = re.search(r'reasoning["\s]*:[\s]*["\']([^"\']+)["\']', response, re.IGNORECASE)
            reasoning = reasoning_match.group(1) if reasoning_match else "No reasoning provided"
            
            # Look for label
            label_match = re.search(r'label["\s]*:[\s]*["\']([^"\']+)["\']', response, re.IGNORECASE)
            label = label_match.group(1) if label_match else "Completely incorrect"
            
            return JudgeScore(
                score=score,
                reasoning=reasoning,
                label=label
            )
            
        except Exception as e:
            print(f"Error in fallback parsing: {e}")
            return JudgeScore(
                score=1,
                reasoning="Error parsing evaluation response",
                label="Completely incorrect"
            )
    
    def batch_evaluate(self, encounters: List[dict]) -> List[JudgeScore]:
        """Evaluate multiple diagnoses in batch for efficiency."""
        results = []
        for encounter in encounters:
            if encounter.get("final_diagnosis") and encounter.get("case_file"):
                score = self.evaluate_diagnosis(
                    encounter["final_diagnosis"],
                    encounter["case_file"]
                )
                results.append(score)
            else:
                results.append(JudgeScore(
                    score=1,
                    reasoning="No diagnosis provided",
                    label="Completely incorrect"
                ))
        return results
