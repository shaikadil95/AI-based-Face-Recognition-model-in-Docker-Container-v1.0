import os
import re
import logging
import pickle

import cv2
import numpy as np
import face_recognition
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# ── Configuration (all values overridable via environment variables) ───────
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
FRAME_SKIP           = int(os.getenv("FRAME_SKIP", "5"))
SCALE_FACTOR         = float(os.getenv("SCALE_FACTOR", "0.25"))
WEBCAM_ID            = int(os.getenv("WEBCAM_ID", "0"))
DISPLAY_VIDEO        = os.getenv("DISPLAY_VIDEO", "false").lower() == "true"
IMAGE_FOLDER         = os.getenv("IMAGE_FOLDER", "Imagefolder")
LOG_FILE             = os.getenv("LOG_FILE", "recognition_log.csv")
MODEL_FILE           = os.getenv("MODEL_FILE", "face_model.pkl")
# Comma-separated list of names allowed to unlock; empty = allow all recognised faces
AUTHORIZED_NAMES     = {n.strip() for n in os.getenv("AUTHORIZED_NAMES", "").split(",") if n.strip()}

# ── Logging ────────────────────────────────────────────────────────────────
_handlers = [logging.StreamHandler()]
if LOG_FILE:
    _handlers.append(logging.FileHandler(LOG_FILE))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=_handlers,
)
log = logging.getLogger(__name__)


class FaceRecognizer:
    """SVM-based face recogniser trained on 128-d dlib face encodings.

    Falls back to nearest-neighbour distance matching when fewer than two
    distinct identities are present in the training set (SVC requires >= 2
    classes).
    """

    def __init__(self):
        self.clf: SVC | None = None
        self.label_encoder: LabelEncoder | None = None
        self.known_encodings: list = []
        self.known_names: list[str] = []

    # ------------------------------------------------------------------
    def train(self, image_folder: str) -> None:
        """Load all images from *image_folder* and fit the classifier.

        Naming convention for training images:
          - Single image : ``Adil.jpg``
          - Multiple images : ``Adil_1.jpg``, ``Adil_2.jpg``, ...
            (the trailing ``_N`` index is stripped to recover the identity)
        """
        encodings: list = []
        names: list[str] = []

        for filename in sorted(os.listdir(image_folder)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            base = os.path.splitext(filename)[0]
            name = re.sub(r"_\d+$", "", base).strip()
            filepath = os.path.join(image_folder, filename)

            try:
                img = face_recognition.load_image_file(filepath)
                enc = face_recognition.face_encodings(img)
                if not enc:
                    log.warning("No face detected in %s — skipped", filename)
                    continue
                encodings.append(enc[0])
                names.append(name)
                log.info("Loaded face for '%s' from %s", name, filename)
            except Exception as exc:
                log.warning("Failed to load %s: %s", filename, exc)

        if not encodings:
            log.error("No usable training images found in '%s'", image_folder)
            return

        self.known_encodings = encodings
        self.known_names = names

        unique = set(names)
        if len(unique) >= 2:
            self.label_encoder = LabelEncoder()
            labels = self.label_encoder.fit_transform(names)
            self.clf = SVC(kernel="rbf", probability=True, C=1.0, gamma="scale")
            self.clf.fit(encodings, labels)
            log.info(
                "SVM trained — %d images, %d identities: %s",
                len(encodings), len(unique), sorted(unique),
            )
        else:
            self.clf = None
            log.info(
                "Only 1 identity found — using distance matching. "
                "Add images for more people to enable SVM."
            )

    # ------------------------------------------------------------------
    def predict(self, encoding: np.ndarray) -> tuple[str, float]:
        """Return *(name, confidence)* for a face encoding.

        Confidence is a probability in [0, 1].  Results below
        CONFIDENCE_THRESHOLD are labelled "Unknown".
        """
        if not self.known_encodings:
            return "Unknown", 0.0

        if self.clf is not None:
            probs = self.clf.predict_proba([encoding])[0]
            idx = int(probs.argmax())
            confidence = float(probs[idx])
            name = self.label_encoder.inverse_transform([idx])[0]
        else:
            distances = face_recognition.face_distance(self.known_encodings, encoding)
            idx = int(distances.argmin())
            confidence = float(1.0 - distances[idx])
            name = self.known_names[idx]

        if confidence < CONFIDENCE_THRESHOLD:
            return "Unknown", confidence
        return name, confidence

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "clf": self.clf,
                    "label_encoder": self.label_encoder,
                    "known_encodings": self.known_encodings,
                    "known_names": self.known_names,
                },
                f,
            )
        log.info("Model saved → %s", path)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.clf            = data["clf"]
        self.label_encoder  = data["label_encoder"]
        self.known_encodings = data["known_encodings"]
        self.known_names    = data["known_names"]
        log.info("Model loaded ← %s", path)


# ── Camera loop ────────────────────────────────────────────────────────────

def run_camera(recognizer: FaceRecognizer) -> None:
    from gpio_lock import DoorLock

    lock = DoorLock()
    cap = cv2.VideoCapture(WEBCAM_ID)
    if not cap.isOpened():
        log.error("Cannot open camera device %d", WEBCAM_ID)
        lock.cleanup()
        return

    log.info(
        "Camera loop started  display=%s  frame_skip=%d  scale=%.2f  threshold=%.2f",
        DISPLAY_VIDEO, FRAME_SKIP, SCALE_FACTOR, CONFIDENCE_THRESHOLD,
    )

    face_locations: list     = []
    face_names: list[str]    = []
    face_confs: list[float]  = []
    frame_count = 0
    inv_scale   = int(round(1.0 / SCALE_FACTOR))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.error("Failed to grab frame — camera disconnected?")
                break

            small     = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
            rgb_small = small[:, :, ::-1]

            if frame_count % FRAME_SKIP == 0:
                face_locations = face_recognition.face_locations(rgb_small)
                encodings      = face_recognition.face_encodings(rgb_small, face_locations)

                face_names, face_confs = [], []
                for enc in encodings:
                    name, conf = recognizer.predict(enc)
                    face_names.append(name)
                    face_confs.append(conf)

                    authorized = name != "Unknown" and (
                        not AUTHORIZED_NAMES or name in AUTHORIZED_NAMES
                    )
                    log.info(
                        "Face: %-20s  conf=%.2f  access=%s",
                        name, conf, "GRANTED" if authorized else "DENIED",
                    )
                    if authorized:
                        lock.unlock()

            frame_count += 1

            if DISPLAY_VIDEO:
                for (top, right, bottom, left), name, conf in zip(
                    face_locations, face_names, face_confs
                ):
                    top    *= inv_scale
                    right  *= inv_scale
                    bottom *= inv_scale
                    left   *= inv_scale

                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    label = f"{name} ({conf:.0%})"
                    cv2.rectangle(
                        frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED
                    )
                    cv2.putText(
                        frame, label, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1,
                    )

                cv2.imshow("Face Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if DISPLAY_VIDEO:
            cv2.destroyAllWindows()
        lock.cleanup()
        log.info("Camera loop stopped")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    recognizer = FaceRecognizer()

    if os.path.exists(MODEL_FILE):
        recognizer.load(MODEL_FILE)
    else:
        recognizer.train(IMAGE_FOLDER)
        recognizer.save(MODEL_FILE)

    if os.getenv("RUN_API", "false").lower() == "true":
        from api import create_app
        flask_app = create_app(recognizer)
        log.info("Starting REST API on port %s", os.getenv("API_PORT", "5000"))
        flask_app.run(
            host="0.0.0.0",
            port=int(os.getenv("API_PORT", "5000")),
            debug=False,
        )
    else:
        run_camera(recognizer)
