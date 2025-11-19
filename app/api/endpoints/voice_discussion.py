import base64
from fastapi import APIRouter, HTTPException
from app.services.voice_service import speech_to_text, text_to_speech
from app.core.schemas import DiscussionIn
from app.services.feedback import discussion_feedback
from pydantic import BaseModel

router = APIRouter(prefix="/voice", tags=["voice_discussion_b64"])

class VoiceDiscussionIn(BaseModel):
    user_id: str = "voice_user"
    session_id: str = "voice_session"
    content: str                     # 뉴스 요약 or 기사
    audio_b64: str                  # Base64 음성 입력
    mode: str = "followup"
    level: str = "beginner"


@router.post("/discussion-b64")
async def voice_discussion_b64(payload: VoiceDiscussionIn):
    """
    (1) Base64 음성 입력 받기
    (2) STT
    (3) AI 토론 수행
    (4) AI 답변을 TTS로 변환
    (5) Base64 MP3로 반환
    """

    try:
        # ---------------------------
        # 1) Base64 → Audio Bytes
        # ---------------------------
        try:
            audio_bytes = base64.b64decode(payload.audio_b64)
        except Exception:
            raise HTTPException(400, "audio_b64 디코딩 실패")

        # ---------------------------
        # 2) STT (음성 → 텍스트)
        # ---------------------------
        stt_text = speech_to_text(audio_bytes)

        # ---------------------------
        # 3) 텍스트 기반 토론 수행
        # ---------------------------
        discussion_input = DiscussionIn(
            user_id=payload.user_id,
            session_id=payload.session_id,
            content=payload.content,
            message=stt_text,
            mode=payload.mode,
            level=payload.level
        )

        result = discussion_feedback(discussion_input)
        ai_reply = result.reply

        # ---------------------------
        # 4) TTS (AI답변 → 음성)
        # ---------------------------
        audio_b64_response = text_to_speech(ai_reply, return_b64=True)

        # ---------------------------
        # 5) 결과 반환
        # ---------------------------
        return {
            "user_input_text": stt_text,
            "ai_reply_text": ai_reply,
            "audio_b64": audio_b64_response,
            "fallback": result.fallback
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice Discussion Error: {e}")
