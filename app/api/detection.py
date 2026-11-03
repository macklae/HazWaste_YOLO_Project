import base64
from functools import lru_cache
from io import BytesIO
from typing import Annotated

import cv2
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

from app.services.inference import YOLOInferenceService

router = APIRouter(prefix="/api", tags=["Detection"])

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_IMAGE_PIXELS = 20_000_000


@lru_cache(maxsize=1)
def get_inference_service():
    """Load the model once and reuse it."""
    return YOLOInferenceService()


def read_uploaded_image(file: UploadFile):
    """Validate and decode JPEG or PNG uploads."""

    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=415,
            detail="Only JPEG and PNG images are supported",
        )

    image_bytes = file.file.read(MAX_IMAGE_SIZE + 1)

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB limit",
        )

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Unable to read the uploaded image",
        )

    expected = "PNG" if file.content_type == "image/png" else "JPEG"

    is_png = image_bytes.startswith(bytes.fromhex("89504e470d0a1a0a"))
    is_jpeg = image_bytes.startswith(bytes.fromhex("ffd8ff"))

    if not ((expected == "PNG" and is_png) or (expected == "JPEG" and is_jpeg)):
        raise HTTPException(
            status_code=400,
            detail="Image content does not match its declared format",
        )

    try:
        with Image.open(BytesIO(image_bytes), formats=["JPEG", "PNG"]) as uploaded:
            if uploaded.format != expected:
                raise HTTPException(
                    status_code=400,
                    detail="Image content does not match its declared format",
                )

            if uploaded.width * uploaded.height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail="Image exceeds the 20 megapixel limit",
                )

            uploaded.load()
            image = uploaded.convert("RGB")

    except HTTPException:
        raise
    except ModuleNotFoundError as exc:
        if exc.name != "pi_heif":
            raise
        raise HTTPException(
            status_code=400,
            detail="Unable to read the uploaded image",
        ) from exc
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail="Unable to read the uploaded image",
        ) from exc

    return image


def encode_png(image):
    """Encode an annotated image as PNG bytes."""

    success, encoded_image = cv2.imencode(".png", image)

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Unable to encode annotated image",
        )

    return encoded_image.tobytes()


@router.post("/detect")
def detect_image(
    file: Annotated[UploadFile, File()],
    confidence: Annotated[float, Form(ge=0.05, le=0.95)] = 0.25,
):
    """Detect hazardous objects in an uploaded image."""

    image = read_uploaded_image(file)

    detections = get_inference_service().predict(
        image,
        confidence=confidence,
    )

    return {
        "filename": file.filename,
        "image_width": image.width,
        "image_height": image.height,
        "detection_count": len(detections),
        "detections": detections,
    }


@router.post("/detect/annotated")
def detect_annotated_image(
    file: Annotated[UploadFile, File()],
    confidence: Annotated[float, Form(ge=0.05, le=0.95)] = 0.25,
):
    """Return an annotated PNG image."""

    image = read_uploaded_image(file)

    annotated_image = get_inference_service().predict_annotated(
        image,
        confidence=confidence,
    )

    return Response(
        content=encode_png(annotated_image),
        media_type="image/png",
    )


@router.post("/detect/combined")
def detect_combined(
    file: Annotated[UploadFile, File()],
    confidence: Annotated[float, Form(ge=0.05, le=0.95)] = 0.25,
):
    """Return detection data and annotated image from one inference."""

    image = read_uploaded_image(file)

    detections, annotated_image = get_inference_service().predict_with_annotation(
        image,
        confidence=confidence,
    )

    annotated_base64 = base64.b64encode(encode_png(annotated_image)).decode("ascii")

    return {
        "filename": file.filename,
        "image_width": image.width,
        "image_height": image.height,
        "detection_count": len(detections),
        "detections": detections,
        "annotated_image": (f"data:image/png;base64,{annotated_base64}"),
    }
