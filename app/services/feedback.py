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
# 요청/명령 감지 기능
# ---------------------------------------------------------
REQUEST_KEYWORDS = [
    "요약", "설명", "정리", "알려줘", "알려 주세요",
    "추천", "분석", "해줘", "해주세요",
    "요약해", "설명해", "정리해","설명해줘", "정리해줘"
]

def is_direct_request(message: str) -> bool:
    msg = message.lower()
    return any(k in msg for k in REQUEST_KEYWORDS)


# ---------------------------------------------------------
# 질문 감지 기능
# ---------------------------------------------------------
QUESTION_KEYWORDS = [
    "뭐야", "뭔데", "뭘까", "왜", "어째서", "어떻게",
    "누구야", "언제야", "어디야",
    "무슨 뜻", "무슨 의미", "어떤 거야", "어떤 건데", "?"
]

def is_question(message: str) -> bool:
    msg = message.lower()
    if msg.endswith("?"):
        return True
    return any(k in msg for k in QUESTION_KEYWORDS)


# ---------------------------------------------------------
# 기사-사용자 메시지 관련성 판단
# ---------------------------------------------------------
def is_relevant_to_content(content: str, message: str) -> bool:
    prompt = (
        "다음 두 문장이 같은 이야기 흐름(context) 또는 주제 영역(topic)에 속하는지 판단하세요. "
        "주제가 완전히 다르다고 확실히 판단되는 경우에만 NO라고 답하고, "
        "내용 흐름이 이어질 가능성이 조금이라도 있으면 YES라고 답하세요.\n\n"
        f"문장1: {content}\n"
        f"문장2: {message}\n\n"
        "정답은 YES 또는 NO만."
    )

    res = run_llm(
        [{"role": "user", "content": prompt}],
        max_tokens=6,
        temperature=0.2
    )
    return res.strip().upper().startswith("Y")


# ---------------------------------------------------------
# 반말/서술체 → 존댓말 변환
# ---------------------------------------------------------
def fix_tone(sentence: str) -> str:
    replacements = {
        "한다.": "합니다.",
        "한다": "합니다",
        "된다.": "됩니다.",
        "된다": "됩니다",
        "있다.": "있습니다.",
        "있다": "있습니다",
        "이다.": "입니다.",
        "이다": "입니다",
        "늘린다": "늘립니다",
        "시행하고 있다": "시행하고 있습니다",
        "추진하고 있다": "추진하고 있습니다"
    }
    for k, v in replacements.items():
        if sentence.endswith(k):
            return sentence[: -len(k)] + v
    return sentence


# ---------------------------------------------------------
# 중복 제거 + 2~3문장 제한
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
        m = re.match(r"^([가-힣a-zA-Z0-9 \"']+?)(은|는)\s", s)
        if m:
            subj = m.group(1).strip()
            if subj in seen_subjects:
                continue
            seen_subjects.add(subj)

        norm = re.sub(r"[^가-힣a-zA-Z ]", " ", s.lower())
        norm = re.sub(r"\s+", " ", norm)
        if not norm:
            continue

        tokens = []
        for t in norm.split():
            if len(t) > 1:
                tokens.append(t)

        noun_set = set(tokens)
        if not noun_set:
            continue

        dup = False
        for prev in meaning_sets:
            if len(noun_set & prev) >= 2:
                dup = True
                break

        if dup:
            continue

        meaning_sets.append(noun_set)
        cleaned.append(fix_tone(s))

        if len(cleaned) >= 3:
            break

    if not cleaned and sentences:
        cleaned = [fix_tone(sentences[0])]

    return " ".join(cleaned[:3]).strip()


# ---------------------------------------------------------
# follow-up 질문 LLM 판단
# ---------------------------------------------------------
def should_followup(message: str, base_reply: str) -> bool:
    prompt = (
        "다음은 사용자 발화와 AI 답변이다.\n"
        "대화를 자연스럽게 이어가기 위해 follow-up 질문을 추가하는 것이 적절한지 판단해라.\n"
        "- 사용자가 자신의 생각/감정/해석을 말한다면 yes\n"
        "- 단순 요청/사실 질문/확인성 발화라면 no\n"
        "- YES 또는 NO만 답해라.\n\n"
        f"[사용자 발화]\n{message}\n\n"
        f"[AI 답변]\n{base_reply}\n\n"
        "답변:"
    )

    res = run_llm(
        [{"role": "user", "content": prompt}],
        max_tokens=3,
        temperature=0.0
    ).lower()

    return "yes" in res


# ---------------------------------------------------------
# Discussion API
# ---------------------------------------------------------
@router.post("/discussion", response_model=DiscussionOut)
def discussion_feedback(payload: DiscussionIn):
    try:
        user_id = payload.user_id
        session_id = payload.session_id
        content = payload.content
        message = payload.message.strip()
        level = payload.level

        # 종료
        if any(w in message.lower() for w in END_WORDS):
            return DiscussionOut(
                reply="좋은 의견 나눠주셔서 감사합니다. 여기서 토론은 마무리할게요.",
                fallback=False, user_id=user_id, session_id=session_id
            )

        # 짧은 입력
        if message in SHORT_INPUTS or len(message) <= 4:
            return DiscussionOut(
                reply=(
                    "조금 더 생각해 보시는 중이신 것 같아요. 기사와 관련해 어떤 점이 가장 마음에 남으셨는지 "
                    "조금만 더 말씀해 주시면 이어서 도와드릴 수 있을 것 같습니다."
                ),
                fallback=False, user_id=user_id, session_id=session_id
            )

        # 답변 생성
        agent = get_agent(user_id, session_id, "")
        chat_res = safe_chat(agent, message)
        base_reply = postprocess_reply(chat_res["answer"].strip())

        
        if not is_relevant_to_content(content, message):
            return DiscussionOut(
                reply="말씀해 주신 내용은 기사 흐름과는 조금 다른 주제인 것 같습니다. 기사 내용이나 그에 대한 생각을 중심으로 이어가 볼까요?",
                fallback=False,
                user_id=user_id,
                session_id=session_id,
            )

        # 요청/질문이면 follow-up 절대 금지
        if is_direct_request(message) or is_question(message):
            final = base_reply

        else:
            # 의견일 때만 follow-up
            if should_followup(message, base_reply):
                q = generate_question(f"{content}\n{message}", mode="followup", level=level)
                q = postprocess_reply(q)
                final = f"{base_reply} {q}".strip()
            else:
                final = base_reply

        return DiscussionOut(
            reply=final,
            fallback=chat_res.get("fallback", False),
            user_id=user_id,
            session_id=session_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Discussion Error: {e}")
