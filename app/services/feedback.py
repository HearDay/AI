from fastapi import APIRouter, HTTPException
from app.modules.question_generator import generate_question
from app.modules.guardrail_service import content_filter, relevance_check
from app.memgpt_agent import get_agent, reset_agent, safe_chat
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

        response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        audio_b64 = base64.b64encode(response.audio_content).decode("utf-8")
        return {"audio_b64": audio_b64}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS 변환 실패: {e}")


# --------------------------------------------------
# 일반 채팅 (MemGPT 기반)
# --------------------------------------------------
@router.post("/chat")
def chat_feedback(payload: dict):
    """
    일반 텍스트 기반 대화 (MemGPT 세션 기억 유지)
    입력: {"user_id": "...", "session_id": "...", "message": "...", "level": "..."}
    출력: {"reply": "..."}
    """
    try:
        user_id = payload.get("user_id", "demo")
        session_id = payload.get("session_id", "default")
        message = payload.get("message", "").strip()
        level = payload.get("level", "beginner")

        if not message:
            raise HTTPException(status_code=400, detail="message 필드가 필요합니다.")

        system_prompt = f"""너는 뉴스 토론 파트너다.
- 사용자의 메시지를 읽고 자연스럽게 응답하라.
- 대화의 맥락을 기억하고 이전 내용과 일관되게 대답하라.
- 필요하면 간단한 피드백 후 탐구형 질문을 덧붙여라."""

        agent = get_agent(user_id, session_id, system_prompt)
        result = safe_chat(agent, message)
        return {
            "reply": result["answer"],
            "fallback": result["fallback"],
            "session_id": session_id,
            "user_id": user_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat Error: {e}")


# --------------------------------------------------
# 토론 (요약 기반, followup/open_question)
# --------------------------------------------------
@router.post("/discussion")
def discussion_feedback(payload: dict):
    """
    뉴스 요약 기반 또는 후속 토론 생성 (MemGPT 세션 포함)
    입력: {"user_id": "...", "session_id": "...", "content": "...", "message": "...", "mode": "...", "level": "..."}
    출력: {"reply": "..."}
    """
    try:
        user_id = payload.get("user_id", "demo")
        session_id = payload.get("session_id", "discussion")
        content = payload.get("content", "")
        message = payload.get("message", "")
        mode = payload.get("mode", "open_question")
        level = payload.get("level", "beginner")

        if not (content or message):
            raise HTTPException(status_code=400, detail="content 또는 message 필드가 필요합니다.")

        context = f"{content}\n\n{message}" if message else content
        question = generate_question(context, mode=mode, level=level)

        is_safe, reason = content_filter(question)
        if not is_safe:
            question = reason

        # MemGPT에 기록 남기기
        agent = get_agent(user_id, session_id, f"토론 모드 ({mode})")
        _ = safe_chat(agent, f"사용자: {message}\n시스템 질문: {question}")

        return {
            "reply": question,
            "mode": mode,
            "level": level,
            "session_id": session_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Discussion Error: {e}")


# --------------------------------------------------
# 세션 초기화 (optional)
# --------------------------------------------------
@router.post("/reset")
def reset_session(payload: dict):
    """특정 사용자 세션의 MemGPT 에이전트 리셋"""
    try:
        user_id = payload.get("user_id", "demo")
        session_id = payload.get("session_id", "default")
        reset_agent(user_id, session_id, "새로운 대화를 시작합니다.")
        return {"ok": True, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset Error: {e}")
