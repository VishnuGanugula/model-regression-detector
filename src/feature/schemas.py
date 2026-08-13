from enum import Enum
from pydantic import BaseModel, Field

class EmailCategory(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    GENERAL = "general"
    UNKNOWN = "unknown" # Fallback if GPT-2 hallucinates

class EmailClassificationResult(BaseModel):
    category: EmailCategory = Field(
        description="The primary category of the customer support email"
    )

class TestCase(BaseModel):
    id: str
    email_text: str
    expected_category: EmailCategory
    expected_difficulty: str = Field(default="normal")
    notes: str = Field(default="")