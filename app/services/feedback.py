import random
from fastapi import APIRouter, HTTPException
from app.services.voice_service import speech_to_text, text_to_speech
from app.services.guardrail_service import content_filter
from app.services.question_generator import generate_question
from app.core.letta_agent import get_agent, safe_chat
from app.core.schemas import DiscussionIn, DiscussionOut
import base64

router = APIRouter(prefix="/feedback", tags=["feedback"])

END_WORDS = [
    "그만", "마칠게", "마칠게요", "종료", "여기까지", "끝",
    "stop", "finish", "이만할게"
]


@router.post("/discussion", response_model=DiscussionOut)
def discussion_feedback(payload: DiscussionIn):
    try:
        user_id = payload.user_id
        session_id = payload.session_id
        content = payload.content
        message = payload.message.strip()
        mode = payload.mode
        level = payload.level

        # 종료 문구 감지
        if any(w in message.lower() for w in END_WORDS):
            return DiscussionOut(
                reply="좋은 의견 나눠주셔서 감사합니다. 여기서 토론은 마무리할게요.",
                fallback=False,
                user_id=user_id,
                session_id=session_id
            )

        # open_question: 첫 질문
        if mode == "open_question":
            question = generate_question(content, mode="open_question", level=level)
            is_safe, reason = content_filter(question)
            if not is_safe:
                question = reason

            return DiscussionOut(
                reply=question,
                fallback=False,
                user_id=user_id,
                session_id=session_id,
            )

        # followup: 자연스러운 의견 + (필요할 경우) 질문
        agent = get_agent(user_id, session_id, "")
        chat_result = safe_chat(agent, message)

        base_reply = chat_result["answer"].strip()

        # followup 문장은 “질문일 수도 있고 아닐 수도 있음”
        context = f"{content}\n\n{message}" if content else message
        question = generate_question(context, mode="followup", level=level)
        is_safe, reason = content_filter(question)
        if not is_safe:
            question = reason

        # question이 실제 질문인지 판별
        if question.endswith("?"):
            final_reply = base_reply + "\n\n" + question
        else:
            final_reply = base_reply

        return DiscussionOut(
            reply=final_reply.strip(),
            fallback=chat_result["fallback"],
            user_id=user_id,
            session_id=session_id,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Discussion Error: {e}")
