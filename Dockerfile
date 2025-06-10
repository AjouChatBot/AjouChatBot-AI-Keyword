FROM ubuntu:22.04

WORKDIR /app

# 시스템 패키지 및 Python 설치
RUN apt-get update && apt-get install -y python3 python3-pip openjdk-11-jdk

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]