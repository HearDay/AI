from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict


# ==============================================
# 뉴스 기반 탐구형 질문 생성 (prompt/question)
# ==============================================
class PromptQuestionIn(BaseModel):
    """뉴스 요약문 기반 탐구형 질문 생성 입력"""
    content: str                     # 뉴스 내용 또는 요약문
    mode: Literal["followup"] = "followup"
    level: Literal["beginner", "intermediate", "advanced"] = "beginner"


class PromptQuestionOut(BaseModel):
    """LLM이 생성한 질문 출력"""
    question: str
    mode: Literal["followup"] = "followup"
    level: Literal["beginner", "intermediate", "advanced"]


# ==============================================
# 토론형 대화 (feedback/discussion)
# ==============================================
class DiscussionIn(BaseModel):
    """AI 토론 입력"""
    user_id: str = "demo_user"           # 사용자 식별자
    session_id: str = "default_session"  # 대화 세션 ID
    content: str                         # 뉴스 내용 또는 기사 일부
    message: str                         # 사용자의 발화 (응답/의견)
    
    mode: Literal["followup"] = "followup"
    level: Literal["beginner", "intermediate", "advanced"] = "beginner"


class DiscussionOut(BaseModel):
    """AI 토론 응답"""
    reply: str                 # 모델이 만든 최종 답변
    fallback: bool = False     
    user_id: str = "demo_user"
    session_id: str = "default_session"
