import sys
import traceback
from pathlib import Path

# Add project root and api dir to sys.path
API_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = API_DIR.parent

for p in [str(PROJECT_ROOT), str(API_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

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

# Expose WSGI application for Vercel
app = app
