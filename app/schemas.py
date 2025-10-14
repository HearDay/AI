from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class FeedbackIn(BaseModel):
    user_id: str = "demo"
    question: str
    user_answer: str
    rubric: Optional[str] = None
    level: Literal["beginner", "intermediate", "advanced"] = "beginner"

class FeedbackOut(BaseModel):
    score: int = Field(ge=0, le=100)
    strengths: List[str]
    improvements: List[str]
    tip: str
    followup: str

class SummarizeIn(BaseModel):
    user_id: str = "demo"
    text: str
    level: Literal["beginner", "intermediate", "advanced"] = "beginner"

class SummarizeOut(BaseModel):
    summary: str
    open_questions: List[str]
    keywords: List[str]

class KeywordsIn(BaseModel):
    text: str
    top_k: int = 8

class KeywordsOut(BaseModel):
    keywords: List[str]
