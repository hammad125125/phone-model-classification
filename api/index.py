"""
Independent REST API Microservice for Mobile Phone Model Classification
Runs as the standalone 'api' service under Vercel Services or local microservice.
"""

import os
import sys
import json
import logging
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Ensure api directory and its src/config are in sys.path
API_DIR = Path(__file__).resolve().parent
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

# Fallback paths inside api directory
MODELS_DIR = API_DIR / "models"
RESULTS_DIR = API_DIR / "results"
WEIGHTS_PATH = MODELS_DIR / "mobile_model.weights.h5"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
MODEL_CONFIG_PATH = MODELS_DIR / "model_config.json"
METRICS_PATH = MODELS_DIR / "metrics.json"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024
TOP_K_PREDICTIONS = 3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ApiService")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app)

# Initialize standalone inference engine
try:
    from src.inference import SmartphoneInferenceEngine
    engine = SmartphoneInferenceEngine(
        weights_path=WEIGHTS_PATH,
        class_names_path=CLASS_NAMES_PATH,
        model_config_path=MODEL_CONFIG_PATH
    )
except Exception as e:
    logger.error(f"Failed to initialize SmartphoneInferenceEngine in API service: {e}")
    engine = None


def get_model_metrics() -> dict:
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return {"evaluated": True, "data": json.load(f)}
        except Exception as e:
            logger.warning(f"Error reading metrics: {e}")
    return {
        "evaluated": False,
        "message": "Evaluation has not yet been conducted on an independent test dataset.",
        "data": None
    }


# Service Root
@app.route("/")
def api_root():
    return jsonify({
        "service": "mobile-model-classification-api",
        "status": "online",
        "endpoints": [
            "/api/health",
            "/api/model-status",
            "/api/classes",
            "/api/metrics",
            "/api/predict"
        ]
    }), 200


# Health Check
@app.route("/health", methods=["GET"])
@app.route("/api/health", methods=["GET"])
def health():
    ready = engine.is_ready if engine else False
    classes_cnt = len(engine.class_names) if engine else 0
    return jsonify({
        "status": "healthy",
        "service": "Mobile Phone Model Classification API Microservice",
        "engine_ready": ready,
        "classes_count": classes_cnt,
        "version": "1.0.0"
    }), 200


# Model Status
@app.route("/model-status", methods=["GET"])
@app.route("/api/model-status", methods=["GET"])
def model_status():
    if not engine:
        return jsonify({"model_ready": False, "error": "Inference engine failed to load"}), 500
    return jsonify(engine.get_status()), 200


# Classes
@app.route("/classes", methods=["GET"])
@app.route("/api/classes", methods=["GET"])
def classes():
    cls_list = engine.class_names if engine else ["Apple_iPhone", "Google_Pixel", "OnePlus", "Samsung_Galaxy", "Xiaomi_Redmi"]
    return jsonify({
        "classes": cls_list,
        "num_classes": len(cls_list)
    }), 200


# Metrics
@app.route("/metrics", methods=["GET"])
@app.route("/api/metrics", methods=["GET"])
def metrics():
    return jsonify(get_model_metrics()), 200


# Prediction
@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def predict():
    file_key = "file" if "file" in request.files else ("image" if "image" in request.files else None)
    if not file_key:
        return jsonify({
            "success": False,
            "error": "No image file provided in request. Expected form key 'file' or 'image'."
        }), 400

    file = request.files[file_key]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename submitted."}), 400

    filename = secure_filename(file.filename)
    file_ext = Path(filename).suffix.lower()

    if file_ext not in ALLOWED_EXTENSIONS:
        return jsonify({
            "success": False,
            "error": f"Unsupported file extension '{file_ext}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }), 400

    top_k = request.form.get("top_k", TOP_K_PREDICTIONS, type=int)

    if not engine or not engine.is_ready:
        if engine:
            engine.reload()
        if not engine or not engine.is_ready:
            return jsonify({
                "success": False,
                "error": "Model weights have not been trained or loaded in this serverless instance.",
                "model_status": "serverless_standby",
                "remediation": "For full GPU deep learning training and inference, run locally via 'python app.py' or connect an external container backend."
            }), 503

    try:
        image_bytes = file.read()
        if len(image_bytes) == 0:
            return jsonify({"success": False, "error": "Submitted file is empty (0 bytes)."}), 400

        prediction_result = engine.predict(image_bytes, top_k=top_k)
        status_code = 200 if prediction_result.get("success") else 422
        return jsonify(prediction_result), status_code

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            "success": False,
            "error": f"Internal inference processing error: {str(e)}"
        }), 500


# Serve Results
@app.route("/results/<path:filename>")
@app.route("/api/results/<path:filename>")
def serve_result(filename):
    return send_from_directory(RESULTS_DIR, filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
