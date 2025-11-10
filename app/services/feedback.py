from fastapi import APIRouter, HTTPException
from app.services.question_generator import generate_question
from app.services.guardrail_service import content_filter
from app.core.memgpt_agent import get_agent, safe_chat
from google.cloud import speech, texttospeech
import base64, os

router = APIRouter(prefix="/feedback", tags=["feedback"])

# --------------------------------------------------
# GCP 인증 (TTS/STT)
# --------------------------------------------------
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "app/core/hearday-4b26b1f78a13.json"


# --------------------------------------------------
# STT (음성 → 텍스트)
# --------------------------------------------------
@router.post("/stt")
def stt_api(payload: dict):
    """Base64 인코딩된 오디오를 받아 텍스트로 변환"""
    try:
        audio_b64 = payload.get("audio_b64")
        if not audio_b64:
            raise HTTPException(status_code=400, detail="audio_b64 필드가 필요합니다.")

        audio_bytes = base64.b64decode(audio_b64)
        client = speech.SpeechClient()
        audio = speech.RecognitionAudio(content=audio_bytes)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code=payload.get("language", "ko-KR"),
        )

        response = client.recognize(config=config, audio=audio)
        if not response.results:
            return {"text": "(인식 실패)"}

        transcript = response.results[0].alternatives[0].transcript
        return {"text": transcript}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT 변환 실패: {e}")


# --------------------------------------------------
# TTS (텍스트 → 음성)
# --------------------------------------------------
@router.post("/tts")
def tts_api(payload: dict):
    """텍스트를 받아 Google Cloud TTS로 Base64 MP3 생성"""
    try:
        text = payload.get("text", "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="text 필드가 비어 있습니다.")

        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="ko-KR",
            name=payload.get("voice", "ko-KR-Neural2-B"),
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        audio_b64 = base64.b64encode(response.audio_content).decode("utf-8")
        return {"audio_b64": audio_b64}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS 변환 실패: {e}")


# --------------------------------------------------
# 토론 (자연스러운 대화 + 탐구형 질문 포함)
# --------------------------------------------------
@router.post("/discussion")
def discussion_feedback(payload: dict):
    """
    대화형 토론 (MemGPT 세션 포함)
    입력:
        {
            "user_id": "...",
            "session_id": "...",
            "content": "...",
            "message": "...",
            "mode": "...",
            "level": "..."
        }
    출력:
        {
            "reply": "...",
            "fallback": bool,
            "session_id": "...",
            "user_id": "..."
        }
    """
    try:
        user_id = payload.get("user_id", "demo")
        session_id = payload.get("session_id", "discussion")
        content = payload.get("content", "")
        message = payload.get("message", "").strip()
        mode = payload.get("mode", "open_question")
        level = payload.get("level", "beginner")

        if not message:
            raise HTTPException(status_code=400, detail="message 필드가 필요합니다.")

        # -------------------------------
        # 시스템 프롬프트 정의
        # -------------------------------
        system_prompt = f"""
        너는 사용자의 토론 파트너이자 대화 상대다.
        - 사용자의 발화를 이해하고 자연스럽게 반응하라.
        - 이전 대화 내용을 기억해 일관성 있게 대화하라.
        - 대화 중 자연스러운 피드백 후 탐구형 질문을 한 문장 덧붙여라.
        - 지나치게 포멀하거나 로봇같이 말하지 말고, 사람처럼 친근하게 표현하라.
        """

        # -------------------------------
        # MemGPT 세션 기반 응답
        # -------------------------------
        agent = get_agent(user_id, session_id, system_prompt)
        chat_result = safe_chat(agent, message)

        # -------------------------------
        # 추가 탐구형 질문 생성 (선택적)
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
        # 응답 조합
        # -------------------------------
        final_reply = chat_result["answer"].strip()
        if question and question not in final_reply:
            final_reply += f"\n\n{question}"

        return {
            "reply": final_reply.strip(),
            "fallback": chat_result["fallback"],
            "session_id": session_id,
            "user_id": user_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Discussion Error: {e}")
