import io
import os
import tempfile
from google.cloud import speech, texttospeech

# GCP 인증 파일 경로 지정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "app/core/hearday-4b26b1f78a13.json"


# STT (음성 -> 텍스트)
def speech_to_text(audio_bytes: bytes) -> str:
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=audio_bytes)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="ko-KR"
    )
    response = client.recognize(config=config, audio=audio)
    if len(response.results) == 0:
        return "(인식 실패)"
    return response.results[0].alternatives[0].transcript


# TTS (텍스트 -> 음성)
def text_to_speech(text: str) -> str:
    """
    Google Cloud TTS 기반
    반환값: mp3 파일 경로(str)
    """
    if not text or text.isspace():
        raise ValueError("텍스트가 비어 있습니다.")

    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="ko-KR",
        name="ko-KR-Neural2-B"
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    # 응답 받기
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # 임시 mp3 파일 생성
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp_file.write(response.audio_content)
    tmp_file.close()

    # 파일 경로 반환
    return tmp_file.name
