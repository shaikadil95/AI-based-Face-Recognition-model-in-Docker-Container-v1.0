import os
import re
import logging

import cv2
import numpy as np
import face_recognition
from flask import Flask, jsonify, request
from flask_cors import CORS

log = logging.getLogger(__name__)

IMAGE_FOLDER     = os.getenv("IMAGE_FOLDER", "Imagefolder")
MODEL_FILE       = os.getenv("MODEL_FILE", "face_model.pkl")
AUTHORIZED_NAMES = {n.strip() for n in os.getenv("AUTHORIZED_NAMES", "").split(",") if n.strip()}


def create_app(recognizer):
    app = Flask(__name__)
    CORS(app)

    @app.route("/status", methods=["GET"])
    def status():
        return jsonify({
            "status": "ok",
            "classifier": "SVM" if recognizer.clf is not None else "distance",
            "known_identities": sorted(set(recognizer.known_names)),
            "confidence_threshold": float(os.getenv("CONFIDENCE_THRESHOLD", "0.60")),
        })

    @app.route("/recognize", methods=["POST"])
    def recognize():
        if "image" not in request.files:
            return jsonify({"error": "Provide an 'image' file in the request"}), 400

        data = np.frombuffer(request.files["image"].read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Could not decode image"}), 400

        rgb = img[:, :, ::-1]
        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, locations)

        results = []
        for enc, (top, right, bottom, left) in zip(encodings, locations):
            name, conf = recognizer.predict(enc)
            authorized = name != "Unknown" and (
                not AUTHORIZED_NAMES or name in AUTHORIZED_NAMES
            )
            log.info(
                "API recognize: %-20s conf=%.2f access=%s",
                name, conf, "GRANTED" if authorized else "DENIED",
            )
            results.append({
                "name": name,
                "confidence": round(float(conf), 3),
                "authorized": authorized,
                "location": {"top": top, "right": right, "bottom": bottom, "left": left},
            })

        return jsonify({"faces": results, "count": len(results)})

    @app.route("/faces", methods=["POST"])
    def add_face():
        if "image" not in request.files or "name" not in request.form:
            return jsonify({"error": "Provide 'image' file and 'name' form field"}), 400

        name = request.form["name"].strip()
        if not name:
            return jsonify({"error": "'name' must not be empty"}), 400

        os.makedirs(IMAGE_FOLDER, exist_ok=True)
        existing = [
            f for f in os.listdir(IMAGE_FOLDER)
            if re.sub(r"_\d+$", "", os.path.splitext(f)[0]).strip() == name
        ]
        idx = len(existing) + 1
        filename = f"{name}_{idx}.jpg"
        filepath = os.path.join(IMAGE_FOLDER, filename)
        request.files["image"].save(filepath)

        recognizer.train(IMAGE_FOLDER)
        recognizer.save(MODEL_FILE)

        log.info("Added face for '%s' as %s, model retrained", name, filename)
        return jsonify({
            "message": f"Added face for '{name}', model retrained",
            "file": filename,
        }), 201

    @app.route("/faces/<name>", methods=["DELETE"])
    def remove_face(name: str):
        removed = []
        for f in os.listdir(IMAGE_FOLDER):
            base = re.sub(r"_\d+$", "", os.path.splitext(f)[0]).strip()
            if base == name:
                os.remove(os.path.join(IMAGE_FOLDER, f))
                removed.append(f)

        if not removed:
            return jsonify({"error": f"No images found for '{name}'"}), 404

        recognizer.train(IMAGE_FOLDER)
        recognizer.save(MODEL_FILE)
        log.info("Removed %d image(s) for '%s', model retrained", len(removed), name)
        return jsonify({
            "message": f"Removed {len(removed)} image(s) for '{name}', model retrained"
        })

    return app
