from app.services.llm_api import run_llm
import re


class HybridMemoryAgent:
    """
    모든 종류의 기사(정치·경제·사회·연예·사건·스포츠·IT·환경 등)에 적용 가능한 토론 Agent.

    - 답변: 한 문단, 2~3문장
    - 기사에 없는 구체 정보 절대 생성 금지
    - 예시/사례/세부 정책·수치 생성 금지 (사용자가 요청한 경우 제외)
    - 의미 기반 중복 제거 (명사 기반)
    - 동일 주어로 시작하는 중복 정의 문장 제거
    """

    def __init__(self, name: str = "discussion_agent"):
        self.name = name
        self.history = []
        print(f"[HybridMemory] initialized: {name}")

    # -----------------------------
    # 메모리
    # -----------------------------
    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def get_recent_messages(self, limit: int = 5):
        return self.history[-limit:]

    # -----------------------------
    # 핵심: 의미 기반 중복 제거 + 예시 삭제 + 주어 중복 제거
    # -----------------------------
    def clean_text(self, text: str) -> str:
        txt = " ".join(text.splitlines()).strip()
        if not txt:
            return txt

        sentences = re.split(r'(?<=[.!?])\s+', txt)
        sentences = [s.strip() for s in sentences if s.strip()]

        cleaned = []
        meaning_sets = []
        seen_subjects = set()   # 동일 주어 중복 방지

        for s in sentences:

            # --------------------------------------
            # (1) 예시 문장은 모든 기사 도메인에서 제거
            # --------------------------------------
            if any(k in s for k in ["예를 들어", "예시로", "예를들어"]):
                continue

            # --------------------------------------
            # (2) 정의 문장의 주어를 추출하고 중복 제거
            #     예: “공공임대주택 공급 확대는 …”
            # --------------------------------------
            m = re.match(r"^([가-힣a-zA-Z0-9 \"']{2,30}?)(은|는)\s", s)

            if m:
                subject = m.group(1).strip()
                if subject in seen_subjects:
                    continue
                seen_subjects.add(subject)

            # --------------------------------------
            # (3) 문장 정규화 → 의미기반 비교용
            # --------------------------------------
            norm = re.sub(r"[^가-힣a-zA-Z ]", " ", s.lower())
            norm = re.sub(r"\s+", " ", norm).strip()

            # --------------------------------------
            # (4) 명사·핵심 단어만 추출
            # --------------------------------------
            tokens = []
            for t in norm.split():
                if len(t) <= 1:
                    continue
                if t.endswith(("다", "요", "니다", "하고", "하며", "되고", "있고")):
                    base = t[:-1]
                    if len(base) > 1:
                        tokens.append(base)
                else:
                    tokens.append(t)

            if not tokens:
                continue

            noun_set = set(tokens)

            # --------------------------------------
            # (5) 의미 중복 판정 (명사 교집합 2개 이상)
            # --------------------------------------
            is_dup = False
            for prev in meaning_sets:
                if len(noun_set & prev) >= 2:
                    is_dup = True
                    break

            if is_dup:
                continue

            meaning_sets.append(noun_set)
            cleaned.append(s)

            if len(cleaned) >= 3:
                break

        # 아무것도 안 남으면 첫 문장 사용
        if not cleaned and sentences:
            cleaned = [sentences[0]]

        return " ".join(cleaned[:3])

    # -----------------------------
    # LLM 대화 (핵심 rules 포함)
    # -----------------------------
    def chat(self, user_input: str, role: str = "user") -> str:
        self.add_message(role, user_input)
        context = self.get_recent_messages()
        conversation = "\n".join(f"{m['role']}: {m['content']}" for m in context)

        # --------------------------------------
        #  시스템 프롬프트
        # --------------------------------------
        system_prompt = (
        "너는 어떤 분야의 기사에도 대응하는 토론 파트너다. "
        "답변은 반드시 공손한 존댓말을 사용하여 한 문단 2~3문장으로 작성한다.\n\n"

        " 절대 금지 규칙:\n"
        "1) 기사에 없는 정책·제도·기술·지원 대상·기간·수치·유형 등을 절대 말하지 않는다.\n"
        "2) 사용자가 기사에 없는 사실 정보를 묻는 경우에만, "
        "'해당 내용은 기사에 언급되지 않아 단정적으로 말씀드리기 어렵습니다.'를 사용한다.\n"
        "3) 사용자가 의견·감정·해석을 말하는 경우에는 이 문구를 절대 사용하지 않는다.\n"
        "4) 의견에는 자연스럽게 공감하거나 조심스럽게 관점을 제시한다.\n"
        "5) 같은 의미를 표현만 바꾸어 반복하지 않는다.\n"
        "6) 단정 표현 금지 ('정책입니다', '제공합니다', '하고 있습니다').\n"
        "7) 사용자가 요청하지 않으면 예시·사례를 들지 않는다.\n"
    )


        raw_reply = run_llm(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": conversation},
            ],
            max_tokens=150,
            temperature=0.3,
        )

        final_reply = self.clean_text(raw_reply)
        self.add_message("assistant", final_reply)
        return final_reply
