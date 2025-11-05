from typing import Dict, Any, Optional

class MemGPTClient:
    #MemGPT 연동 래퍼(환경에 맞춰 실제 호출부 연결).
    #현재는 더미 메서드만 제공함
    

    def __init__(self):
        pass

    def write_event(self, user_id: str, event: str, meta: Optional[Dict[str,Any]] = None) -> None:
        # 실제 연동 필요 시 구현
        pass

    def fetch_context(self, user_id: str) -> str:
        # 필요 시 컨텍스트 반환 로직 구현
        return ""
