# AI-based Face Recognition in Docker Container

Real-time face recognition system deployed in a Docker container, targeting IoT edge devices (Raspberry Pi). Recognised faces can trigger a GPIO-controlled door lock, implementing the access-control use-case described in the thesis below.

> **Thesis:** Shaik, A. & Chetlur, U. V. (2020). *Design and implementation of an AI-based Face Recognition model in Docker Container on IoT Platform*. Master's thesis, Blekinge Institute of Technology, Faculty of Computing.
> DiVA record: [diva2:1457000](https://www.diva-portal.org/smash/record.jsf?pid=diva2:1457000)
> Full text: [FULLTEXT01.pdf](https://www.diva-portal.org/smash/get/diva2:1457000/FULLTEXT01.pdf)

> **Note:** This is a revised version of the original prototype submitted with the thesis.
> The original code (84-line `face.py`, Python 3.4 Dockerfile) is preserved at commit [`aaf1031`](https://github.com/shaikadil95/AI-based-Face-Recognition-model-in-Docker-Container-v1.0/tree/aaf1031).
> This revision adds the SVM classifier, Flask REST API, GPIO door lock, headless mode, and multi-arch Docker support described in the thesis but absent from the original implementation.

---

## Architecture

```
Imagefolder/              Training
  Adil_1.jpg  ──┐
  Adil_2.jpg  ──┤──► FaceRecognizer.train()
  kurt_1.jpg  ──┘        │
                     128-d dlib encodings
                          │
                     SVC (RBF kernel)   ← scikit-learn SVM
                          │
Camera / CSI ──► frame downscale ──► face_locations ──► predict()
                                                            │
                                              name + confidence
                                                            │
                                          ┌─────────────────┤
                                          ▼                 ▼
                                     GPIO relay        REST API
                                    (door lock)      /recognize
```

---

## Features

| Feature | Details |
|---|---|
| **Classifier** | SVM (RBF kernel, `predict_proba`) via scikit-learn; falls back to nearest-neighbour distance for single-identity sets |
| **Confidence threshold** | Configurable; faces below threshold labelled *Unknown* |
| **Multiple training images** | `Adil_1.jpg`, `Adil_2.jpg` → all mapped to identity *Adil* |
| **GPIO door lock** | Active-low relay on configurable BCM pin; auto-relocks after timeout |
| **Simulation mode** | GPIO gracefully degrades on non-Pi hardware |
| **REST API** | Flask endpoints for remote image recognition, face management, status |
| **Headless operation** | `DISPLAY_VIDEO=false` (default) for server/IoT deployment |
| **Audit log** | Every recognition event written to `recognition_log.csv` |
| **Multi-arch Docker** | `python:3.11-slim`; builds on x86-64 and ARM (Raspberry Pi) |

---

## Training data

Place images in `Imagefolder/`. Two naming conventions are supported:

```
Imagefolder/
  Adil.jpg          ← single image for identity "Adil"
  kurt.jpg          ← single image for identity "kurt"
  Adil_1.jpg        ← first of multiple images for "Adil"
  Adil_2.jpg        ← second image — trailing _N is stripped automatically
```

Supported formats: `.jpg`, `.jpeg`, `.png`.  
Each image must contain exactly one face. Add at least **5–10 images per person** for reliable SVM accuracy.

The model is trained at startup and cached to `face_model.pkl`. Delete the pickle file to force a retrain after updating the image folder.

---

## Configuration

All parameters are set via environment variables so they can be overridden at runtime without rebuilding the image.

| Variable | Default | Description |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.60` | Minimum SVM probability to accept a match |
| `FRAME_SKIP` | `5` | Process every Nth frame (performance/latency trade-off) |
| `SCALE_FACTOR` | `0.25` | Downscale factor before detection (smaller = faster) |
| `WEBCAM_ID` | `0` | OpenCV camera index |
| `DISPLAY_VIDEO` | `false` | Show annotated video window (requires X11) |
| `IMAGE_FOLDER` | `Imagefolder` | Path to training images |
| `MODEL_FILE` | `face_model.pkl` | Cached SVM model path |
| `LOG_FILE` | `recognition_log.csv` | Audit log path |
| `AUTHORIZED_NAMES` | *(empty)* | Comma-separated whitelist; empty = allow all recognised faces |
| `GPIO_LOCK_PIN` | `17` | BCM GPIO pin connected to lock relay |
| `UNLOCK_DURATION` | `3.0` | Seconds the door stays unlocked |
| `RUN_API` | `false` | Set `true` to start the REST API instead of the camera loop |
| `API_PORT` | `5000` | Flask listening port |

---

## Quick start

### Docker (recommended)

```bash
# Build
docker build -t face-recognition .

# Camera mode (headless, GPIO simulation on non-Pi)
docker run --rm \
  --device /dev/video0 \
  -e AUTHORIZED_NAMES="Adil,kurt" \
  face-recognition

# API mode
docker run --rm -p 5000:5000 -e RUN_API=true face-recognition
```

### Docker Compose

```bash
# Camera mode
docker compose up face-recognition

# API mode
docker compose up face-api
```

### Supported platforms

| Platform | Target | Notes |
|---|---|---|
| `linux/amd64` | x86-64 | Laptops, desktops, Docker Desktop on Intel Mac |
| `linux/arm64` | 64-bit ARM | **Apple M1/M2/M3 Mac**, Raspberry Pi 4 (64-bit OS) |
| `linux/arm/v7` | 32-bit ARM | Raspberry Pi 2 / 3 / 4 (32-bit Raspbian) — **primary target** |

The image runs on any platform. Features degrade gracefully when hardware is unavailable:

| Feature | Raspberry Pi | Any other machine |
|---|---|---|
| Face recognition | Full | Full |
| REST API | Full | Full |
| Camera loop | Full (webcam / CSI) | Full (webcam required) |
| GPIO door lock | Real relay on configured pin | Simulated — logged but no pin toggle |

**Apple M1/M2/M3:** Docker Desktop runs containers as `linux/arm64` automatically. Just `docker build` and `docker run` as usual — no extra flags needed.

### Build for your platform

```bash
# Standard build (Docker auto-detects your platform)
docker build -t face-recognition .

# Explicitly target a platform (useful for cross-compilation)
docker build --platform linux/arm64   -t face-recognition:arm64 .   # M1/M2 Mac, Pi 64-bit
docker build --platform linux/arm/v7  -t face-recognition:armv7 .   # Pi 32-bit
docker build --platform linux/amd64   -t face-recognition:amd64 .   # Intel/AMD
```

### Build all platforms at once (buildx)

```bash
# One-time setup
docker buildx create --use --name multiarch

# Build all three platforms into a single multi-arch manifest
docker buildx build \
  --platform linux/amd64,linux/arm64,linux/arm/v7 \
  -t yourname/face-recognition:latest \
  --push .
```

### Manual (development)

```bash
pip install -r requirements.txt
# Optional on Raspberry Pi:
pip install RPi.GPIO==0.7.1

# Camera mode
python face.py

# API mode
RUN_API=true python face.py
```

---

## REST API

All endpoints accept and return JSON. Start the server with `RUN_API=true`.

### `GET /status`

```bash
curl http://localhost:5000/status
```
```json
{
  "status": "ok",
  "classifier": "SVM",
  "known_identities": ["Adil", "kurt"],
  "confidence_threshold": 0.6
}
```

### `POST /recognize`

Submit an image for recognition. Returns all detected faces with name, confidence, and access decision.

```bash
curl -X POST http://localhost:5000/recognize \
  -F "image=@photo.jpg"
```
```json
{
  "count": 1,
  "faces": [
    {
      "name": "Adil",
      "confidence": 0.91,
      "authorized": true,
      "location": {"top": 42, "right": 180, "bottom": 130, "left": 92}
    }
  ]
}
```

### `POST /faces`  — add a training image

```bash
curl -X POST http://localhost:5000/faces \
  -F "name=Alice" \
  -F "image=@alice.jpg"
```
```json
{"message": "Added face for 'Alice', model retrained", "file": "Alice_1.jpg"}
```

### `DELETE /faces/<name>`  — remove a person

```bash
curl -X DELETE http://localhost:5000/faces/Alice
```
```json
{"message": "Removed 2 image(s) for 'Alice', model retrained"}
```

---

## Raspberry Pi deployment

1. Flash Raspberry Pi OS Lite and enable the camera and SSH.
2. Install Docker: `curl -fsSL https://get.docker.com | sh`
3. Clone this repo and copy your training images into `Imagefolder/`.
4. Set `GPIO_LOCK_PIN` to the BCM pin connected to your relay module.
5. Run:

```bash
docker compose up face-recognition
```

The lock relay wiring convention is **active-low**: the pin is held HIGH (locked) and pulled LOW to unlock for `UNLOCK_DURATION` seconds.

---

## Repository structure

```
.
├── face.py            # FaceRecognizer class + camera loop + entry point
├── api.py             # Flask REST API (create_app factory)
├── gpio_lock.py       # GPIO door lock with simulation fallback
├── requirements.txt   # Python dependencies
├── Dockerfile         # Multi-arch container build
├── docker-compose.yml # Camera and API service definitions
└── Imagefolder/       # Training images (add your own)
    ├── Adil.jpg
    └── kurt.jpg
```

---

## Citation

If you use this work, please cite the thesis:

```bibtex
@mastersthesis{shaik2020face,
  author  = {Shaik, Adil and Chetlur, Uma Vidyadhari},
  title   = {Design and implementation of an AI-based Face Recognition model
             in Docker Container on IoT Platform},
  school  = {Blekinge Institute of Technology},
  year    = {2020},
  url     = {https://www.diva-portal.org/smash/record.jsf?pid=diva2:1457000}
}
```
