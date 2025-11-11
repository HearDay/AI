import os
import tempfile
import base64
from google.cloud import speech, texttospeech

# --------------------------------------------------
# GCP 인증 경로 설정
# --------------------------------------------------
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "app/core/hearday-4b26b1f78a13.json"


# --------------------------------------------------
# STT (음성 → 텍스트)
# --------------------------------------------------
def speech_to_text(audio_bytes: bytes) -> str:
    """음성 데이터를 받아 텍스트로 변환"""
    try:
        client = speech.SpeechClient()
        audio = speech.RecognitionAudio(content=audio_bytes)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="ko-KR",
        )
        response = client.recognize(config=config, audio=audio)

        if not response.results:
            return "(인식 실패)"
        return response.results[0].alternatives[0].transcript

    except Exception as e:
        print(f"[voice_service:STT] Error: {e}")
        return "(오디오 인식 중 오류 발생)"


# --------------------------------------------------
# TTS (텍스트 → 음성, Base64 인코딩 지원)
# --------------------------------------------------
def text_to_speech(text: str, return_b64: bool = False) -> str:
    """
    텍스트를 음성으로 변환
    - return_b64=True: Base64 문자열 반환 (API 통신용)
    - return_b64=False: 임시 파일 경로 반환 (내부 테스트용)
    """
    try:
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="ko-KR",
            name="ko-KR-Neural2-B"
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        if return_b64:
            # Base64로 반환
            return base64.b64encode(response.audio_content).decode("utf-8")

        # 임시 파일 저장
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp_file.write(response.audio_content)
        tmp_file.close()
        return tmp_file.name

    except Exception as e:
        print(f"[voice_service:TTS] Error: {e}")
        return ""
