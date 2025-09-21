# ===== Base =====
FROM python:3.12-slim


RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt




COPY . /app

ENV PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8080 \
    STORAGE_DIR=/app/storage \
    MODEL_PATH=best.pt \
    CONF_TH=0.5 \
    IMG_SIZE=640 \
    WINDOW=10 \
    K_FALLEN=6 \
    K_STANDING=6 \
    NO_STAND_TIMEOUT_S=20 \
    MAX_RECENT_EVENTS=100


RUN mkdir -p /app/storage


EXPOSE 8080


CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --proxy-headers --forwarded-allow-ips '*' "]