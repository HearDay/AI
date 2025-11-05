from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict

# 피드백 / 요약 / 키워드 스키마

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

# 토론용 AI api
class PromptQuestionIn(BaseModel):
    discussionId: Optional[int] = None
    nickname: Optional[str] = None
    articleTitle: Optional[str] = None
    content: str
    previousMessages: Optional[List[Dict[str, str]]] = []
    message: Optional[str] = None
    mode: Literal["open_question", "followup"] = "open_question"
    level: Literal["beginner", "intermediate", "advanced"] = "beginner"

class PromptQuestionOut(BaseModel):
    question: str
    mode: str
    level: str
