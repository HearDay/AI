from fastapi import APIRouter, HTTPException
from app.services.voice_service import speech_to_text, text_to_speech
from app.services.question_generator import generate_question
from app.services.guardrail_service import content_filter
from app.core.letta_agent import get_agent, safe_chat
from app.core.schemas import DiscussionIn, DiscussionOut
import base64

router = APIRouter(prefix="/feedback", tags=["feedback"])


# --------------------------------------------------
# STT (음성 → 텍스트)
# --------------------------------------------------
@router.post("/stt")
def stt_api(payload: dict):
    """Base64 인코딩된 오디오를 받아 텍스트로 변환"""
    audio_b64 = payload.get("audio_b64")
    if not audio_b64:
        raise HTTPException(status_code=400, detail="audio_b64 필드가 필요합니다.")

    try:
        audio_bytes = base64.b64decode(audio_b64)
        text = speech_to_text(audio_bytes)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT 변환 실패: {e}")


# --------------------------------------------------
# TTS (텍스트 → 음성)
# --------------------------------------------------
@router.post("/tts")
def tts_api(payload: dict):
    """텍스트를 받아 Google Cloud TTS로 Base64 MP3 생성"""
    text = payload.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text 필드가 비어 있습니다.")

    try:
        audio_b64 = text_to_speech(text, return_b64=True)
        return {"audio_b64": audio_b64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS 변환 실패: {e}")


# --------------------------------------------------
# 토론 (자연스러운 대화 + 피드백 + 탐구형 질문 포함)
# --------------------------------------------------
@router.post("/discussion", response_model=DiscussionOut)
def discussion_feedback(payload: DiscussionIn):
    """
    대화형 토론 (letta 세션 포함)
    입력:
        DiscussionIn 스키마 참고
    출력:
        DiscussionOut 스키마 참고
    """
    try:
        # -------------------------------
        # 입력값 언패킹
        # -------------------------------
        user_id = payload.user_id
        session_id = payload.session_id
        content = payload.content
        message = payload.message.strip()
        mode = payload.mode
        level = payload.level

        # -------------------------------
        # 시스템 프롬프트 정의 (대답 + 후속질문 생성 지시)
        # -------------------------------
        system_prompt = f"""
        너는 사용자의 한국인 뉴스토론 파트너이자 대화 상대다.
        - 사용자의 발화를 이해하고 공감하는 짧은 피드백을 먼저 해라.
        - 이어서 자연스럽게 후속 질문을 덧붙여라.
        - 이전 대화 내용을 기억해 일관성 있게 대화해라.
        - 지나치게 포멀하거나 로봇같이 말하지 말고, 사람처럼 자연스럽게 표현해라.
        """

        # -------------------------------
        # letta 세션 기반 응답
        # -------------------------------
        agent = get_agent(user_id, session_id, system_prompt)
        chat_result = safe_chat(agent, message)

        # -------------------------------
        # 탐구형 질문 추가 (선택적, 필터 포함)
        # -------------------------------
        context = f"{content}\n\n{message}" if content else message
        try:
            question = generate_question(context, mode=mode, level=level)
            is_safe, reason = content_filter(question)
            if not is_safe:
                question = reason
        except Exception:
            question = ""

        # -------------------------------
        # 최종 응답 조합
        # -------------------------------
        base_reply = chat_result["answer"].strip()
        final_reply = base_reply
        if question and question not in base_reply:
            final_reply += f"\n\n{question}"

        # -------------------------------
        # 스키마에 맞게 반환
        # -------------------------------
        return DiscussionOut(
            reply=final_reply.strip(),
            fallback=chat_result["fallback"],
            user_id=user_id,
            session_id=session_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Discussion Error: {e}")
