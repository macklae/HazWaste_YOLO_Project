from io import BytesIO
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.detection import get_inference_service
from app.main import app


@pytest.fixture
def mock_inference():
    """Replace the real YOLO model with a mock during tests."""
    service = Mock()

    detections = [
        {
            "class_id": 0,
            "class_name": "Shock Absorber",
            "confidence": 0.85,
            "bbox": [10.0, 20.0, 100.0, 150.0],
        }
    ]

    service.predict.return_value = detections
    service.predict_annotated.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    service.predict_with_annotation.return_value = (
        detections,
        np.zeros((100, 100, 3), dtype=np.uint8),
    )

    app.dependency_overrides[get_inference_service] = lambda: service

    # The existing endpoints call get_inference_service() directly,
    # so patch the function in the detection module.
    from unittest.mock import patch

    with patch(
        "app.api.detection.get_inference_service",
        return_value=service,
    ):
        yield service

    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


def make_image_bytes():
    image = Image.new("RGB", (100, 100), color="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_endpoint(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_dashboard_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_detect_success(client, mock_inference):
    response = client.post(
        "/api/detect",
        files={"file": ("test.png", make_image_bytes(), "image/png")},
        data={"confidence": "0.5"},
    )

    assert response.status_code == 200

    data = response.json()
    assert data["detection_count"] == 1
    assert data["detections"][0]["class_name"] == "Shock Absorber"
    mock_inference.predict.assert_called_once()


def test_annotated_image_success(client, mock_inference):
    response = client.post(
        "/api/detect/annotated",
        files={"file": ("test.png", make_image_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")
    mock_inference.predict_annotated.assert_called_once()


def test_combined_detection(client, mock_inference):
    response = client.post(
        "/api/detect/combined",
        files={"file": ("test.png", make_image_bytes(), "image/png")},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["detection_count"] == 1
    assert len(data["detections"]) == 1
    assert data["annotated_image"].startswith("data:image/png;base64,")

    mock_inference.predict_with_annotation.assert_called_once()


def test_invalid_file_type(client):
    response = client.post(
        "/api/detect",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 415


def test_oversized_image(client):
    oversized_content = b"x" * (10 * 1024 * 1024 + 1)

    response = client.post(
        "/api/detect",
        files={"file": ("large.png", oversized_content, "image/png")},
    )

    assert response.status_code == 413


@pytest.mark.parametrize("confidence", ["0.01", "1.0", "invalid"])
def test_invalid_confidence(client, confidence):
    response = client.post(
        "/api/detect",
        files={"file": ("test.png", make_image_bytes(), "image/png")},
        data={"confidence": confidence},
    )

    assert response.status_code == 422


def test_corrupted_image(client):
    response = client.post(
        "/api/detect",
        files={"file": ("broken.png", b"invalid image content", "image/png")},
    )

    assert response.status_code == 400
