
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from threading import Lock
from time import sleep
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api import detection
from app.main import app
from app.services.inference import YOLOInferenceService


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_model(monkeypatch):
    service = Mock()
    service.predict.return_value = []
    service.predict_with_annotation.return_value = (
        [],
        __import__("numpy").zeros((20, 20, 3), dtype="uint8"),
    )
    monkeypatch.setattr(detection, "get_inference_service", lambda: service)
    return service


def image_bytes(format_name="PNG", size=(100, 100), mode="RGB"):
    image = Image.new(mode, size, color="white")
    buffer = BytesIO()
    image.save(buffer, format=format_name)
    return buffer.getvalue()


def upload(client, content, content_type="image/png", endpoint="/api/detect"):
    return client.post(
        endpoint,
        files={"file": ("test-image", content, content_type)},
    )


def test_empty_image_returns_400(client):
    response = upload(client, b"")
    assert response.status_code == 400


@pytest.mark.parametrize(
    ("declared_type", "actual_format"),
    [
        ("image/png", "JPEG"),
        ("image/jpeg", "PNG"),
    ],
)
def test_mismatched_image_format_returns_400(
    client, declared_type, actual_format
):
    content = image_bytes(actual_format)
    response = upload(client, content, declared_type)
    assert response.status_code == 400


def test_image_over_20_megapixels_returns_413(client):
    content = image_bytes(size=(5000, 4200), mode="1")
    response = upload(client, content)
    assert response.status_code == 413


def test_valid_jpeg_reaches_model(client, mock_model):
    content = image_bytes(format_name="JPEG")
    response = upload(client, content, "image/jpeg")
    assert response.status_code == 200
    mock_model.predict.assert_called_once()


def test_combined_endpoint_uses_one_inference_call(client, mock_model):
    content = image_bytes()
    response = upload(
        client,
        content,
        endpoint="/api/detect/combined",
    )
    assert response.status_code == 200
    mock_model.predict_with_annotation.assert_called_once()
    mock_model.predict.assert_not_called()


def test_concurrent_inference_is_serialized():
    service = YOLOInferenceService.__new__(YOLOInferenceService)
    service._predict_lock = Lock()

    active = 0
    maximum_active = 0
    counter_lock = Lock()

    def fake_predict(**kwargs):
        nonlocal active, maximum_active

        with counter_lock:
            active += 1
            maximum_active = max(maximum_active, active)

        try:
            sleep(0.03)
            return ["fake-result"]
        finally:
            with counter_lock:
                active -= 1

    service.model = Mock()
    service.model.predict.side_effect = fake_predict

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(
                service._run_inference,
                ["image"] * 8,
            )
        )

    assert results == ["fake-result"] * 8
    assert service.model.predict.call_count == 8
    assert maximum_active == 1
