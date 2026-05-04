FROM python:3.11-slim

# BuildKit automatically sets these when using --platform or docker buildx.
# Declaring them here makes them available in RUN steps.
ARG TARGETPLATFORM
ARG TARGETARCH
ARG BUILDPLATFORM
RUN echo "Building for $TARGETPLATFORM on $BUILDPLATFORM"

# cmake + build-essential are required by dlib when no pre-built wheel is
# available for the target platform (e.g. linux/arm/v7).
# libopenblas-dev / liblapack-dev cover all supported architectures in Debian slim.
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

# Install RPi.GPIO on ARM targets (arm = linux/arm/v7, arm64 = linux/arm64).
# We use $TARGETARCH (set by BuildKit) instead of uname -m so this works
# correctly during cross-compilation on an x86 build host.
# gpio_lock.py silently falls back to simulation mode on non-Pi hardware.
RUN if [ "$TARGETARCH" = "arm" ] || [ "$TARGETARCH" = "arm64" ]; then \
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
