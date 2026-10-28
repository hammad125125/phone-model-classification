import sys
import traceback
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from app import app
except Exception as e:
    from flask import Flask, jsonify
    err_tb = traceback.format_exc()
    app = Flask(__name__)
    
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def error_fallback(path):
        return jsonify({
            "success": False,
            "error": "Serverless Initialization Error",
            "message": str(e),
            "traceback": err_tb
        }), 500

# Vercel entrypoint
app = app
