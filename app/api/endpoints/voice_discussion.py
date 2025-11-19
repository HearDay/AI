from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from app.services.voice_service import speech_to_text, text_to_speech
from app.core.schemas import DiscussionIn
from app.services.feedback import discussion_feedback

router = APIRouter(prefix="/voice", tags=["voice_discussion"])

@router.post("/discussion")
async def voice_discussion(
    user_id: str = Form("voice_user"),
    session_id: str = Form("voice_session"),
    content: str = Form(...),
    mode: str = Form("followup"),
    level: str = Form("beginner"),
    audio: UploadFile = File(...)
):
    """
    (1) 음성 받아서 STT
    (2) AI 토론 수행
    (3) TTS로 변환한 Base64 반환
    """

    try:
        # ---------------------------
        # 1) 음성 파일 → bytes 로 읽기
        # ---------------------------
        audio_bytes = await audio.read()

        # ---------------------------
        # 2) STT (음성 → 텍스트)
        # ---------------------------
        stt_text = speech_to_text(audio_bytes)

        # ---------------------------
        # 3) AI 토론 (텍스트 기반)
        # ---------------------------
        payload = DiscussionIn(
            user_id=user_id,
            session_id=session_id,
            content=content,
            message=stt_text,
            mode=mode,
            level=level
        )

        discussion_result = discussion_feedback(payload)
        ai_reply = discussion_result.reply

        # ---------------------------
        # 4) TTS 변환 (AI 답변 → Base64 음성)
        # ---------------------------
        audio_b64 = text_to_speech(ai_reply, return_b64=True)

        # ---------------------------
        # 5) 결과 반환
        # ---------------------------
        return {
            "user_input_text": stt_text,
            "ai_reply_text": ai_reply,
            "audio_b64": audio_b64,
            "fallback": discussion_result.fallback
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice Discussion Error: {e}")
