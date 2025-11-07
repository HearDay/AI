# 1. 베이스 이미지: Python 3.12 Slim 버전을 사용합니다.
FROM python:3.12-slim

# 2. 작업 디렉토리: 컨테이너 내부의 /app 폴더를 작업 공간으로 지정합니다.
WORKDIR /app

# 3. 의존성 설치:
# 먼저 requirements.txt 파일만 복사합니다.
COPY requirements.txt .

# pip를 업그레이드하고 requirements.txt에 있는 모든 라이브러리를 설치합니다.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. 소스 코드 복사:
# 현재 폴더(.)의 모든 파일(app/, .env 등)을 컨테이너의 /app 폴더로 복사합니다.
COPY . .

# 5. 포트 개방:
# Uvicorn이 8000번 포트에서 실행될 것이므로, 컨테이너의 8000번 포트를 엽니다.
EXPOSE 8000

# 6. 서버 실행 명령어:
# 컨테이너가 시작될 때 이 명령어를 실행합니다.
# [중요!] "--host 0.0.0.0"은 컨테이너 외부에서 접속을 허용하기 위해 필수입니다.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]