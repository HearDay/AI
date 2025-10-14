# app/modules/open_questions.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from ..config import settings
from ..prompt_templates import build_open_question_prompt, conversational_rewrite

# 전역 초기화 제거하고, 지연 로딩으로 전환
_tokenizer = None
_model = None

def _load_llm():
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model
    print(f"🔹 Loading model (lazy): {settings.HF_MODEL_ID}")
    _tokenizer = AutoTokenizer.from_pretrained(settings.HF_MODEL_ID, use_fast=True)
    _model = AutoModelForCausalLM.from_pretrained(
        settings.HF_MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=settings.DEVICE_MAP,
    )
    return _tokenizer, _model

def _apply_chat_template(system: str, user: str) -> str:
    tok, _ = _load_llm()
    if hasattr(tok, "apply_chat_template") and tok.chat_template:
        messages = [{"role":"system","content":system},{"role":"user","content":user}]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"<<SYS>>\n{system}\n<</SYS>>\n\n[USER]\n{user}\n[/USER]\n[ASSISTANT] "

@torch.inference_mode()
def _generate(prompt: str) -> str:
    tok, mdl = _load_llm()
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    output_ids = mdl.generate(
        **inputs,
        max_new_tokens=settings.HF_MAX_NEW_TOKENS,
        do_sample=True,
        temperature=settings.HF_TEMPERATURE,
        top_p=settings.HF_TOP_P,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    text = tok.decode(output_ids[0], skip_special_tokens=True)
    return text.split(prompt, 1)[-1].strip()

def generate_open_questions(summary: str, level: str = "beginner") -> list[str]:
    user_prompt = build_open_question_prompt(summary, level)
    system_prompt = "너는 뉴스 토론을 돕는 한국어 어시스턴트다. 사용자가 편하게 느끼는 자연스러운 대화 톤을 유지한다."
    full_prompt = _apply_chat_template(system_prompt, user_prompt)

    raw = _generate(full_prompt)
    raw = conversational_rewrite(raw)

    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    qs = []
    for l in lines:
        if l[:2].isdigit() and ")" in l:
            qs.append(l.split(")", 1)[-1].strip(" :"))
        elif l and l[0].isdigit() and l[1] in (")", ".", "）"):
            qs.append(l[2:].strip(" :"))
        if len(qs) == 3: break
    if len(qs) < 3:
        fallback = [
            "지금 이 사안에서 가장 핵심적인 사실은 무엇이라고 보시나요?",
            "이 결정이 누구에게 어떤 변화를 만들 수 있을까요?",
            "당신이라면 어떤 대안을 먼저 시도해 보시겠어요?"
        ]
        qs = (qs + fallback)[:3]
    qs = [q.rstrip("?.") + "?" if not q.endswith("?") else q for q in qs]
    return qs

@app.post("/prompt/open_questions")
def prompt_open_questions(level: str = Form("beginner"), summary: str = Form(...)):
    try:
        questions = generate_open_questions(summary, level)
    except Exception as e:
        return {"level": level, "open_questions": [], "error": str(e)}
    return {"level": level, "open_questions": questions}
