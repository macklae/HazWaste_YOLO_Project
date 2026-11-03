
from datetime import datetime
from pathlib import Path
import re
import shutil

ROOT = Path.cwd()

FILES = [
    "app/api/detection.py",
    "app/services/inference.py",
    "app/static/js/camera.js",
    "app/static/js/upload.js",
]


def replace_exact(text, old, new, name):
    if text.count(old) != 1:
        raise ValueError(f"{name}: expected code not found exactly once.")
    return text.replace(old, new, 1)


def replace_function(text, name, replacement):
    pattern = rf"(?ms)^def {name}\(.*?(?=^def |^@router\.|^class |\Z)"
    updated, count = re.subn(
        pattern, lambda match: replacement + "\n\n", text, count=1
    )
    if count != 1:
        raise ValueError(f"Cannot locate function: {name}")
    return updated


def main():
    originals = {}

    for relative in FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing file: {relative}")
        originals[relative] = path.read_text()

    changes = originals.copy()

    detection = changes["app/api/detection.py"]

    detection = replace_exact(
        detection,
        "MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB",
        "MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB\n"
        "MAX_IMAGE_PIXELS = 20_000_000",
        "Image size configuration",
    )

    new_reader = '''
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

    if not (
        (expected == "PNG" and is_png)
        or (expected == "JPEG" and is_jpeg)
    ):
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
'''.strip()

    detection = replace_function(
        detection, "read_uploaded_image", new_reader
    )
    changes["app/api/detection.py"] = detection

    inference = changes["app/services/inference.py"]

    inference = replace_exact(
        inference,
        "from pathlib import Path",
        "from pathlib import Path\nfrom threading import Lock",
        "Inference imports",
    )

    inference = replace_exact(
        inference,
        "        self.model = YOLO(str(MODEL_PATH))",
        "        self.model = YOLO(str(MODEL_PATH))\n"
        "        self._predict_lock = Lock()",
        "Model initialization",
    )

    pattern = (
        r"(?ms)^    def _run_inference\(.*?"
        r"(?=^    @staticmethod|^    def |\Z)"
    )

    replacement = '''    def _run_inference(self, image, confidence=0.25):
        with self._predict_lock:
            results = self.model.predict(
                source=image,
                conf=confidence,
                imgsz=640,
                verbose=False,
            )
            return results[0]

'''

    inference, count = re.subn(
        pattern, lambda match: replacement, inference, count=1
    )

    if count != 1:
        raise ValueError("Inference function not found.")

    changes["app/services/inference.py"] = inference

    camera = changes["app/static/js/camera.js"]

    camera = replace_exact(
        camera,
        '''      cameraOverlay.width = cameraVideo.videoWidth;
      cameraOverlay.height = cameraVideo.videoHeight;''',
        '''      if (
        cameraOverlay.width !== cameraVideo.videoWidth ||
        cameraOverlay.height !== cameraVideo.videoHeight
      ) {
        cameraOverlay.width = cameraVideo.videoWidth;
        cameraOverlay.height = cameraVideo.videoHeight;
      }''',
        "Camera overlay",
    )

    camera = replace_exact(
        camera,
        '''    } finally {
      liveDetectionAbort = null;
      scheduleNextFrame(session);''',
        '''    } finally {
      if (session === liveDetectionSession) {
        liveDetectionAbort = null;
      }
      scheduleNextFrame(session);''',
        "Camera cleanup",
    )

    changes["app/static/js/camera.js"] = camera

    upload = changes["app/static/js/upload.js"]

    upload = replace_exact(
        upload,
        '''      console.error("Upload detection error:", error);
      setStatus(error.message);''',
        '''      console.error("Upload detection error:", error);
      resetResults();
      setStatus(error.message);''',
        "Upload error handling",
    )

    changes["app/static/js/upload.js"] = upload

    for relative, content in changes.items():
        if relative.endswith(".py"):
            compile(content, relative, "exec")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = ROOT / f".hazwaste-backup-{timestamp}"
    backup.mkdir()

    for relative in FILES:
        destination = backup / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)

    try:
        for relative, content in changes.items():
            (ROOT / relative).write_text(content)
    except Exception:
        for relative in FILES:
            shutil.copy2(backup / relative, ROOT / relative)
        raise

    print("Update applied successfully.")
    print("Backup:", backup)
    print("Next: run automated tests.")


if __name__ == "__main__":
    main()
