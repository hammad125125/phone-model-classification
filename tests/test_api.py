"""
Integration Tests for Flask Web Application & REST API Endpoints
"""

import io
import pytest
from PIL import Image
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestAPIEndpoints:
    def test_health_endpoint(self, client):
        """Tests that GET /api/health returns HTTP 200 and expected diagnostic keys."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "healthy"
        assert "engine_ready" in data
        assert "classes_count" in data

    def test_model_status_endpoint(self, client):
        """Tests that GET /api/model-status returns structured model information."""
        response = client.get("/api/model-status")
        assert response.status_code == 200
        data = response.get_json()
        assert "model_ready" in data
        assert "architecture" in data
        assert "input_dimensions" in data

    def test_predict_endpoint_missing_file(self, client):
        """Tests that POST /api/predict without a file returns 400 Bad Request or 503 if untrained."""
        response = client.post("/api/predict", data={})
        # If untrained returns 503, if trained returns 400
        assert response.status_code in (400, 503)
        data = response.get_json()
        assert data["success"] is False

    def test_predict_endpoint_invalid_file_extension(self, client):
        """Tests that POST /api/predict with a non-image file (.txt) is rejected."""
        data = {
            "file": (io.BytesIO(b"dummy text content"), "test_file.txt")
        }
        response = client.post("/api/predict", data=data, content_type="multipart/form-data")
        assert response.status_code in (400, 503)
        res_json = response.get_json()
        assert res_json["success"] is False

    def test_pages_render(self, client):
        """Verifies that core HTML pages render successfully without 500 errors."""
        for path in ["/", "/demo", "/dataset", "/model", "/docs", "/resources"]:
            response = client.get(path)
            assert response.status_code == 200, f"Page {path} failed with status {response.status_code}"

