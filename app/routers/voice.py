from fastapi import APIRouter, File, UploadFile, HTTPException, Response
from app.modules.voice_service import speech_to_text, text_to_speech

router = APIRouter(prefix="/audio", tags=["voice"])

# STT (음성 -> 텍스트)
@router.post("/stt")
async def convert_audio_to_text(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        text = speech_to_text(audio_bytes)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT Error: {e}")

# TTS (텍스트 -> 음성)
@router.post("/tts")
async def convert_text_to_audio(text: str):
    try:
        audio_bytes = text_to_speech(text)
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS Error: {e}")
