from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.modules.question_generator import generate_question
from app.modules.voice_service import text_to_speech
import json
import asyncio
import base64

router = APIRouter(prefix="/ws", tags=["discussion"])


@router.websocket("/discussion")
async def discussion_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket 연결됨")

    try:
        while True:
            # 클라이언트로부터 데이터 수신
            data = await websocket.receive_text()
            payload = json.loads(data)

            # 메시지 파싱
            user_message = payload.get("message", "")
            level = payload.get("level", "beginner")

            # LLM 호출 (질문 생성 or 후속 대화)
            llm_reply = generate_question(user_message, mode="followup", level=level)

            # Google Cloud TTS 변환 (mp3 파일 경로 반환)
            tts_path = text_to_speech(llm_reply)

            # mp3 파일 읽어서 Base64 인코딩
            with open(tts_path, "rb") as f:
                tts_base64 = base64.b64encode(f.read()).decode("utf-8")

            # 클라이언트로 보낼 응답 데이터
            response = {
                "type": "reply",
                "reply": llm_reply,
                "tts_audio": tts_base64  # 클라이언트에서 바로 재생 가능
            }

            # JSON 직렬화 후 WebSocket으로 전송
            await websocket.send_text(json.dumps(response))

            # 서버 안정화용 짧은 대기
            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        print("WebSocket 연결 종료됨")
        await websocket.close()
