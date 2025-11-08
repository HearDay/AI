FROM python:3.12-slim
WORKDIR /app

# 1. requirements만 먼저 복사 (캐시 최적화)
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 2. 실제 코드만 복사
COPY app ./app
COPY main.py ./

# 3. 환경 변수 / 실행 설정
ENV PYTHONUNBUFFERED=1 TZ=Asia/Seoul

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]