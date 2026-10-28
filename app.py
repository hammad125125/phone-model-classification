"""
Flask Web Application and REST API Service
Mobile Phone Model Classification Using Deep Learning and Transfer Learning
"""

import os
import sys
import json
import logging
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    MODELS_DIR,
    DATASET_DIR,
    RESULTS_DIR,
    SAMPLE_IMAGES_DIR,
    WEIGHTS_PATH,
    CLASS_NAMES_PATH,
    MODEL_CONFIG_PATH,
    TRAINING_CONFIG_PATH,
    METRICS_PATH,
    DATASET_SUMMARY_PATH,
    CONFUSION_MATRIX_PNG,
    TRAINING_CURVES_PNG,
    ALLOWED_EXTENSIONS,
    MAX_CONTENT_LENGTH,
    FLASK_HOST,
    FLASK_PORT,
    FLASK_DEBUG,
    TOP_K_PREDICTIONS
)
from src.inference import SmartphoneInferenceEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("WebApp")

app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=str(PROJECT_ROOT / "static")
)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "academic-deep-learning-secret-key-2026")
CORS(app)

INFERENCE_API_URL = os.environ.get("INFERENCE_API_URL", "").rstrip("/")

# Initialize in-memory singleton inference engine
engine = SmartphoneInferenceEngine(
    weights_path=WEIGHTS_PATH,
    class_names_path=CLASS_NAMES_PATH,
    model_config_path=MODEL_CONFIG_PATH
)


def get_dataset_summary() -> dict:
    """Reads dataset_summary.json or scans dataset if available."""
    if DATASET_SUMMARY_PATH.exists():
        try:
            with open(DATASET_SUMMARY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    default_classes = engine.class_names if engine.is_ready else ["Apple_iPhone", "Google_Pixel", "OnePlus", "Samsung_Galaxy", "Xiaomi_Redmi"]
    return {
        "status": "unscanned",
        "classes": default_classes,
        "num_classes": len(default_classes),
        "splits": {
            "train": {"total": 0, "per_class": {cls: 0 for cls in default_classes}},
            "validation": {"total": 0, "per_class": {cls: 0 for cls in default_classes}},
            "test": {"total": 0, "per_class": {cls: 0 for cls in default_classes}}
        },
        "total_images": 0,
        "leakage_warnings": []
    }


def get_model_metrics() -> dict:
    """Reads genuine evaluation metrics from models/metrics.json or returns pending state."""
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


# ---------------------------------------------------------------------------
# Web Pages
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    """Home page with project introduction, hero banner, and key highlights."""
    status = engine.get_status()
    metrics = get_model_metrics()
    return render_template("index.html", status=status, metrics=metrics)


@app.route("/demo")
def demo():
    """Interactive demo page: drag-and-drop upload, real-time prediction, confidence display."""
    status = engine.get_status()
    sample_images = []
    if SAMPLE_IMAGES_DIR.exists():
        sample_images = [f.name for f in SAMPLE_IMAGES_DIR.iterdir() if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS]
    return render_template("demo.html", status=status, sample_images=sample_images)


@app.route("/dataset")
def dataset_page():
    """Dataset information, structure, licensing, class distribution, and split breakdown."""
    summary = get_dataset_summary()
    return render_template("dataset.html", summary=summary)


@app.route("/model")
def model_page():
    """Model information: MobileNetV2 architecture, transfer learning details, metrics."""
    status = engine.get_status()
    metrics = get_model_metrics()
    model_cfg = {}
    if MODEL_CONFIG_PATH.exists():
        try:
            with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
                model_cfg = json.load(f)
        except Exception:
            pass
    return render_template("model.html", status=status, metrics=metrics, model_cfg=model_cfg)


@app.route("/docs")
def docs_page():
    """Project documentation: problem statement, methodology, architecture, and limitations."""
    return render_template("docs.html")


@app.route("/resources")
def resources_page():
    """Project resources: GitHub repo, report outline, presentation outline, citation."""
    return render_template("resources.html")


# ---------------------------------------------------------------------------
# Static Artifact Servicing
# ---------------------------------------------------------------------------

@app.route("/results/<path:filename>")
def serve_result_artifact(filename):
    """Serves generated confusion matrix or training curves safely."""
    return send_from_directory(RESULTS_DIR, filename)


@app.route("/dataset/sample_test_images/<path:filename>")
def serve_sample_image(filename):
    """Serves sample test images for demo preview."""
    return send_from_directory(SAMPLE_IMAGES_DIR, filename)


# ---------------------------------------------------------------------------
# REST API Endpoints
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def api_health():
    """Health check endpoint providing runtime and system diagnostic information."""
    return jsonify({
        "status": "healthy",
        "service": "Mobile Phone Model Classification API",
        "engine_ready": engine.is_ready,
        "classes_count": len(engine.class_names),
        "version": "1.0.0"
    }), 200


@app.route("/api/model-status", methods=["GET"])
def api_model_status():
    """Returns detailed status of the model weights and loaded classes."""
    return jsonify(engine.get_status()), 200


@app.route("/api/classes", methods=["GET"])
def api_classes():
    """Returns list of currently configured/trained classes."""
    return jsonify({
        "classes": engine.class_names,
        "num_classes": len(engine.class_names)
    }), 200


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Returns genuine evaluation metrics if available."""
    metrics_info = get_model_metrics()
    return jsonify(metrics_info), 200


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Secure prediction endpoint.
    Expects 'file' or 'image' multipart form-data.
    Validates MIME type, file extension, and file size before calling the inference engine.
    Supports proxying to an external containerized backend via INFERENCE_API_URL (ideal for Vercel).
    """
    # 0. Check for file presence
    file_key = "file" if "file" in request.files else ("image" if "image" in request.files else None)
    if not file_key:
        return jsonify({
            "success": False,
            "error": "No image file provided in request. Expected form key 'file' or 'image'."
        }), 400

    file = request.files[file_key]
    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "Empty filename submitted."
        }), 400

    filename = secure_filename(file.filename)
    file_ext = Path(filename).suffix.lower()

    # Validate file extension
    if file_ext not in ALLOWED_EXTENSIONS:
        return jsonify({
            "success": False,
            "error": f"Unsupported file extension '{file_ext}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }), 400

    top_k = request.form.get("top_k", TOP_K_PREDICTIONS, type=int)

    # 1. If external backend is specified (e.g. for Vercel serverless -> container backend)
    if INFERENCE_API_URL:
        try:
            import requests
            image_bytes = file.read()
            files = {file_key: (filename, image_bytes, file.content_type or "image/jpeg")}
            data = {"top_k": top_k}
            resp = requests.post(f"{INFERENCE_API_URL}/api/predict", files=files, data=data, timeout=30)
            return jsonify(resp.json()), resp.status_code
        except Exception as e:
            logger.error(f"Inference proxy error to {INFERENCE_API_URL}: {e}")
            return jsonify({
                "success": False,
                "error": f"Failed to connect to backend inference service: {str(e)}"
            }), 502

    # 2. Local in-memory inference check
    if not engine.is_ready:
        engine.reload()
        if not engine.is_ready:
            return jsonify({
                "success": False,
                "error": "Model weights have not been trained or loaded.",
                "model_status": "untrained",
                "remediation": "For serverless deployments (Vercel), configure INFERENCE_API_URL pointing to your Docker/Render backend, or run locally via 'python app.py'."
            }), 503

    # 3. Read bytes and execute local inference
    try:
        image_bytes = file.read()
        if len(image_bytes) == 0:
            return jsonify({"success": False, "error": "Submitted file is empty (0 bytes)."}), 400

        prediction_result = engine.predict(image_bytes, top_k=top_k)
        status_code = 200 if prediction_result.get("success") else 422
        return jsonify(prediction_result), status_code

    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        return jsonify({
            "success": False,
            "error": "An internal error occurred while processing the image."
        }), 500


# ---------------------------------------------------------------------------
# Error Handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found_error(error):
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": "Endpoint not found"}), 404
    return render_template("base.html", body_content="<div class='container my-5 text-center'><h2>404 - Page Not Found</h2><a href='/' class='btn btn-primary mt-3'>Return Home</a></div>"), 404


@app.errorhandler(413)
def request_entity_too_large(error):
    msg = f"Uploaded file exceeds maximum permitted size of {MAX_CONTENT_LENGTH // (1024*1024)}MB."
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": msg}), 413
    return f"<h3>File Too Large</h3><p>{msg}</p>", 413


@app.errorhandler(500)
def internal_server_error(error):
    import traceback
    tb = traceback.format_exc()
    logger.error(f"500 Internal Error: {tb}")
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": str(error), "traceback": tb}), 500
    return f"""
    <div style="font-family:sans-serif; padding:2rem; max-width:800px; margin:auto;">
      <h2 style="color:#ef4444;">Application Error (500)</h2>
      <p><strong>{error}</strong></p>
      <pre style="background:#f1f5f9; padding:1rem; border-radius:6px; overflow:auto;">{tb}</pre>
      <a href="/" style="display:inline-block; margin-top:1rem; color:#2563eb;">Return to Home</a>
    </div>
    """, 500


if __name__ == "__main__":
    logger.info(f"Starting Flask server on http://{FLASK_HOST}:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
