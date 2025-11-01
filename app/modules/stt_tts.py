import tempfile
from gtts import gTTS
import speech_recognition as sr


def speech_to_text(audio_path: str) -> str:
    """
    음성 -> 텍스트 (STT)
    Google SpeechRecognition 로컬 방식 사용
    
    """
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio, language="ko-KR")
        return text.strip()
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        return ""


def text_to_speech(text: str, output_path: str = None) -> str:
    """
     텍스트 -> 음성 (TTS)
    Google TTS (gTTS) 기반
    반환값: mp3 파일 경로

    """
    if not text or text.isspace():
        raise ValueError("텍스트가 비어 있습니다. 변환할 수 없습니다.")

    if not output_path:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        output_path = tmp.name

    tts = gTTS(text=text, lang="ko")
    tts.save(output_path)
    return output_path
