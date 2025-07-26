FROM python:3.10-slim AS build-env

RUN apt-get update && apt-get install -y git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY *.py ./

COPY logging_config.json ./

CMD ["python3", "main.py"]
