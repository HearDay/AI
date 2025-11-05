from fastapi import APIRouter, HTTPException
from app.modules.question_generator import generate_question
from app.modules.voice_service import text_to_speech
from app.modules.guardrail_service import content_filter, relevance_check
import base64
import asyncio

router = APIRouter(prefix="/discussion", tags=["discussion"])

@router.post("")
async def discussion_endpoint(payload: dict):
    """
    실시간 토론 (REST API 버전)
    - 입력: {"message": "...", "level": "..."}
    - 출력: {"reply": "...", "tts_audio": "..."}
    """
    try:
        # 요청 파라미터 추출
        user_message = payload.get("message", "")
        level = payload.get("level", "beginner")

        if not user_message:
            raise HTTPException(status_code=400, detail="message 필드가 필요합니다.")

        # LLM 호출 (후속 질문 생성)
        llm_reply = generate_question(user_message, mode="followup", level=level)

        # 가드레일 필터 적용
        is_safe, reason = content_filter(llm_reply)
        if not is_safe:
            llm_reply = reason

        # 관련성 검증
        if not relevance_check(user_message, llm_reply):
            llm_reply += " (⚠️ 주제와 관련이 적은 응답일 수 있습니다.)"

        # Google Cloud TTS 변환 (bytes 반환)
        tts_bytes = text_to_speech(llm_reply)

        # Base64 인코딩 (클라이언트가 바로 재생 가능하도록)
        tts_base64 = base64.b64encode(tts_bytes).decode("utf-8")

        # 응답 JSON
        response = {
            "reply": llm_reply,
            "tts_audio": tts_base64
        }

        await asyncio.sleep(0.1)
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
