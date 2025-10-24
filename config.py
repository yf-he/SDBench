"""Configuration settings for SDBench."""

import os
from typing import Optional

class Config:
    """Configuration class for SDBench."""
    
    # OpenAI API settings
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # Model settings
    GATEKEEPER_MODEL: str = "gpt-4o-mini"
    JUDGE_MODEL: str = "gpt-4o"
    
    # Cost settings
    PHYSICIAN_VISIT_COST: float = 300.0
    
    # Evaluation settings
    CORRECT_DIAGNOSIS_THRESHOLD: int = 4  # Score >= 4 is considered correct
    
    # Data settings
    VALIDATION_CASES: int = 248
    TEST_CASES: int = 56
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        return True
