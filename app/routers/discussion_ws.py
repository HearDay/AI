from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.modules.question_generator import generate_question
from app.modules.voice_service import text_to_speech
from app.modules.guardrail_service import content_filter, relevance_check
import json
import asyncio
import base64
import os

router = APIRouter(prefix="/ws", tags=["discussion"])


@router.websocket("/discussion")
async def discussion_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket 연결됨")

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)

            user_message = payload.get("message", "")
            level = payload.get("level", "beginner")

            # LLM 호출
            llm_reply = generate_question(user_message, mode="followup", level=level)

            # 가드레일 필터
            is_safe, reason = content_filter(llm_reply)
            if not is_safe:
                llm_reply = reason

            # 관련성 검증
            if not relevance_check(user_message, llm_reply):
                llm_reply += " (⚠️ 주제와 관련이 적은 응답일 수 있습니다.)"

            # Google TTS 변환 (mp3 파일 경로 반환)
            tts_path = text_to_speech(llm_reply)

            # 파일 읽어서 Base64 변환
            with open(tts_path, "rb") as f:
                tts_base64 = base64.b64encode(f.read()).decode("utf-8")

            response = {
                "type": "reply",
                "reply": llm_reply,
                "tts_audio": tts_base64
            }

            await websocket.send_text(json.dumps(response))
            await asyncio.sleep(0.1)

            # 임시 파일 삭제 (테스트용입니다)
            try:
                os.remove(tts_path)
            except Exception:
                pass

    except WebSocketDisconnect:
        print("WebSocket 연결 종료됨")
        await websocket.close()
