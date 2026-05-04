FROM python:3.11-slim

# cmake + build-essential are required by dlib when no pre-built wheel is
# available for the target platform (e.g. Raspberry Pi armv7l / aarch64).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install RPi.GPIO only when running on ARM (Raspberry Pi hardware).
# The gpio_lock module silently falls back to simulation mode when the
# package is absent, so this is a soft dependency.
RUN if [ "$(uname -m)" = "armv7l" ] || [ "$(uname -m)" = "aarch64" ]; then \
        pip install --no-cache-dir RPi.GPIO==0.7.1; \
    fi

COPY Imagefolder /app/Imagefolder/
COPY face.py api.py gpio_lock.py /app/

# ── Default environment (override at runtime with -e or docker-compose) ───
ENV DISPLAY_VIDEO=false \
    CONFIDENCE_THRESHOLD=0.60 \
    FRAME_SKIP=5 \
    SCALE_FACTOR=0.25 \
    WEBCAM_ID=0 \
    IMAGE_FOLDER=/app/Imagefolder \
    LOG_FILE=/app/recognition_log.csv \
    MODEL_FILE=/app/face_model.pkl \
    GPIO_LOCK_PIN=17 \
    UNLOCK_DURATION=3.0 \
    RUN_API=false \
    API_PORT=5000 \
    AUTHORIZED_NAMES=""

EXPOSE 5000

CMD ["python", "face.py"]
