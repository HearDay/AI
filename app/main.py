from fastapi import FastAPI, Form
from .prompt_templates import build_open_question_prompt
from .modules.open_questions import generate_open_questions

app = FastAPI(title="Debate Prompt & Open Questions", version="0.1.0")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/prompt/preview")
def prompt_preview(level: str = Form("beginner"), summary: str = Form(...)):
    return {"level": level, "prompt": build_open_question_prompt(summary, level)}

@app.post("/prompt/open_questions")
def prompt_open_questions(level: str = Form("beginner"), summary: str = Form(...)):
    return {"level": level, "open_questions": generate_open_questions(summary, level)}
