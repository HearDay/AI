from app.services.question_generator import generate_question

if __name__ == "__main__":
    print("[시작] Kanana 모델 기반 질문 생성 테스트 중...")

    context = "AI가 의료 분야에서 의사 역할을 대체할 수 있을지 논란이 되고 있다."
    
    # 1️⃣ 개방형 질문 테스트
    print("[단계1] open_question 실행 중...")
    result_open = generate_question(context, mode="open", level="beginner")
    print("=== [OPEN QUESTION] ===")
    print(result_open)
    print()

    # 2️⃣ 후속 질문 테스트
    print("[단계2] followup 실행 중...")
    user_reply = "저는 AI가 의사를 완전히 대체하긴 어렵다고 봐요."
    result_follow = generate_question(user_reply, mode="followup", level="beginner")
    print("=== [FOLLOW-UP QUESTION] ===")
    print(result_follow)

    print("[완료] 테스트 종료.")
