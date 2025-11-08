from app.services.question_generator import generate_question

print("[시작] Mock 기반 AI 토론 테스트 중...\n")

# 뉴스 요약 입력
context = "최근 정부가 AI 의료 보조 시스템의 법적 책임 구조를 논의 중이다."

# Step 1: 개방형 질문
open_q = generate_question(context, mode="open_question", level="beginner")
print("[AI 첫 질문]")
print(open_q)

# Step 2: 사용자 답변 (가상 입력)
user_reply = "AI가 의사를 완전히 대체하는건 불가능하다고 생각해."
print("\n[사용자 답변]")
print(user_reply)

# Step 3: 후속 질문
followup_q = generate_question(user_reply, mode="followup", level="beginner")
print("\n[AI 후속 질문]")
print(followup_q)

print("\n[테스트 종료]")
