"""Data models for SDBench."""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from enum import Enum

class ActionType(str, Enum):
    """Types of actions a diagnostic agent can take."""
    ASK_QUESTIONS = "ask_questions"
    REQUEST_TESTS = "request_tests"
    DIAGNOSE = "diagnose"

class AgentAction(BaseModel):
    """An action taken by the diagnostic agent."""
    action_type: ActionType
    content: str
    timestamp: Optional[float] = None

class CaseFile(BaseModel):
    """Complete case file with all information."""
    case_id: str
    initial_abstract: str
    full_case_text: str
    ground_truth_diagnosis: str
    publication_year: int
    is_test_case: bool = False

class GatekeeperResponse(BaseModel):
    """Response from the gatekeeper agent."""
    response_text: str
    is_synthetic: bool = False
    cost: Optional[float] = None

class JudgeScore(BaseModel):
    """Score from the judge agent."""
    score: int = Field(ge=1, le=5)
    reasoning: str
    label: str

class DiagnosticEncounter(BaseModel):
    """A complete diagnostic encounter."""
    case_id: str
    actions: List[AgentAction] = []
    gatekeeper_responses: List[GatekeeperResponse] = []
    total_cost: float = 0.0
    final_diagnosis: Optional[str] = None
    judge_score: Optional[JudgeScore] = None
    is_complete: bool = False

class BenchmarkResult(BaseModel):
    """Results from running the benchmark."""
    diagnostic_accuracy: float
    average_cost: float
    total_cases: int
    correct_cases: int
    encounter_results: List[DiagnosticEncounter]

class CPTMapping(BaseModel):
    """CPT code mapping for cost estimation."""
    test_name: str
    cpt_codes: List[str]
    estimated_cost: float
    confidence: float
