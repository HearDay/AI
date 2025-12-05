from fastapi import APIRouter, HTTPException
from app.services.guardrail_service import content_filter
from app.services.question_generator import generate_question
from app.core.letta_agent import get_agent, safe_chat
from app.core.schemas import DiscussionIn, DiscussionOut
from app.services.llm_api import run_llm
import re

router = APIRouter(prefix="/feedback", tags=["feedback"])

END_WORDS = [
    "그만", "마칠게", "마칠게요", "종료", "여기까지", "끝",
    "stop", "finish", "이만할게"
]

SHORT_INPUTS = [
    "음", "음.", "음..", "음...", "흠", "흠...",
    "글쎄", "글쎄요", "잘 모르겠어", "모르겠네"
]


# ---------------------------------------------------------
# 기사와 사용자 발화의 관련성 판정 
# ---------------------------------------------------------
def is_relevant_to_content(content: str, message: str) -> bool:
    """
    YES 경향 — 주제가 조금이라도 이어지면 YES
    완전히 다른 경우만 NO
    """
    prompt = (
        "다음 두 문장이 같은 이야기 흐름(context) 또는 주제 영역(topic)에 속하는지 판단해라.\n"
        "주제가 완전히 다르다고 확실하게 판단되는 경우에만 NO라고 답해라.\n"
        "내용 흐름이 이어질 가능성이 조금이라도 있으면 YES라고 답해라.\n\n"
        f"문장1: {content}\n"
        f"문장2: {message}\n\n"
        "정답은 YES 또는 NO만."
    )

    res = run_llm([{"role": "user", "content": prompt}], max_tokens=6, temperature=0.2)
    return res.strip().upper().startswith("Y")


# ---------------------------------------------------------
# follow-up 질문 필요 여부 판단 (LLM 판단)
# ---------------------------------------------------------
def should_followup(message: str, base_reply: str) -> bool:
    """
    대화 맥락 상 follow-up 질문이 자연스러운지 yes/no로 판단.
    """
    prompt = (
        "다음은 사용자 발화와 AI 답변이다.\n"
        "대화를 자연스럽게 이어가기 위해 follow-up 질문을 추가하는 것이 적절한지 판단해라.\n\n"
        "기준:\n"
        "- 사용자가 자신의 생각이나 감정을 더 말할 여지가 있으면 yes\n"
        "- 사용자 질문이 이미 충분히 해결되었거나, 마무리 느낌이면 no\n"
        "- yes 또는 no만 답해라.\n\n"
        f"[사용자]\n{message}\n\n"
        f"[AI]\n{base_reply}\n\n"
        "답변:"
    )

    reply = run_llm(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": message}
        ],
        max_tokens=3,
        temperature=0.0,
    ).strip().lower()

    return "yes" in reply


# ---------------------------------------------------------
# 중복 제거 + 2~3문장 제한 후처리
# ---------------------------------------------------------
def postprocess_reply(text: str) -> str:
    if not text:
        return text

    txt = " ".join(text.splitlines()).strip()
    sentences = re.split(r'(?<=[.!?])\s+', txt)
    sentences = [s.strip() for s in sentences if s.strip()]

    cleaned = []
    meaning_sets = []
    seen_subjects = set()

    for s in sentences:
        # 동일 주어(…은/는) 중복 정의 제거
        m = re.match(r"^([가-힣a-zA-Z0-9 \"']+?)(은|는)\s", s)
        if m:
            subject = m.group(1).strip()
            if subject in seen_subjects:
                continue
            seen_subjects.add(subject)

        # 의미 중복 제거 (명사 기반)
        norm = re.sub(r"[^가-힣a-zA-Z ]", " ", s.lower())
        norm = re.sub(r"\s+", " ", norm).strip()

        if not norm:
            continue

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

        # 중복 의미 판정
        dup = False
        for prev in meaning_sets:
            if len(noun_set & prev) >= 2:
                dup = True
                break
        if dup:
            continue

        meaning_sets.append(noun_set)
        cleaned.append(s)

        if len(cleaned) >= 3:
            break

    if not cleaned and sentences:
        cleaned = [sentences[0]]

    return " ".join(cleaned[:3]).strip()


# ---------------------------------------------------------
# Discussion API (최종본)
# ---------------------------------------------------------
@router.post("/discussion", response_model=DiscussionOut)
def discussion_feedback(payload: DiscussionIn):
    try:
        user_id = payload.user_id
        session_id = payload.session_id
        content = payload.content
        message = payload.message.strip()
        mode = payload.mode
        level = payload.level

        # 종료 감지
        if any(w in message.lower() for w in END_WORDS):
            reply = "좋은 의견 나눠주셔서 감사합니다. 여기서 토론은 마무리할게요."
            return DiscussionOut(reply=reply, fallback=False, user_id=user_id, session_id=session_id)

        # 첫 질문
        if mode == "open_question":
            question = generate_question(content, mode="open_question", level=level)
            question = postprocess_reply(question)
            return DiscussionOut(reply=question, fallback=False, user_id=user_id, session_id=session_id)

        # 짧은 입력 대응
        if message in SHORT_INPUTS or len(message) <= 4:
            reply = (
                "조금 고민 중이신 것 같네요. 기사 내용과 관련해 어떤 점이 마음에 남으셨는지 "
                "조금만 더 말씀해 주시면 대화를 이어서 도와드릴 수 있을 것 같습니다."
            )
            return DiscussionOut(reply=reply, fallback=False, user_id=user_id, session_id=session_id)

        # follow-up
        agent = get_agent(user_id, session_id, "")
        chat_result = safe_chat(agent, message)
        base_reply = postprocess_reply(chat_result["answer"].strip())

        # -------------------------------
        # 기사 관련성 판단 
        # -------------------------------
        if not is_relevant_to_content(content, message):
            return DiscussionOut(
                reply="말씀해 주신 내용은 지금 이야기 중인 기사와는 조금 거리가 있어 보여요. 기사와 관련된 생각이나 궁금한 점이 있다면 다시 말씀해 주세요.",
                fallback=False,
                user_id=user_id,
                session_id=session_id,
            )

        # follow-up 필요 여부 판단
        if should_followup(message, base_reply):
            follow_q = generate_question(f"{content}\n{message}", mode="followup", level=level)
            follow_q = postprocess_reply(follow_q)
            final_reply = f"{base_reply} {follow_q}".strip()
        else:
            final_reply = base_reply

        return DiscussionOut(
            reply=final_reply,
            fallback=chat_result.get("fallback", False),
            user_id=user_id,
            session_id=session_id,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Discussion Error: {e}")
