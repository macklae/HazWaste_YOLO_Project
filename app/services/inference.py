from pathlib import Path
from threading import Lock

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "yolo26n_original_best.pt"


class YOLOInferenceService:
    def __init__(self):
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")

        self.model = YOLO(str(MODEL_PATH))
        self._predict_lock = Lock()

    def _run_inference(self, image, confidence=0.25):
        with self._predict_lock:
            results = self.model.predict(
                source=image,
                conf=confidence,
                imgsz=640,
                verbose=False,
            )
            return results[0]

    @staticmethod
    def _extract_detections(result):
        """Convert YOLO predictions into structured detection data."""
        detections = []

        for box in result.boxes:
            class_id = int(box.cls.item())

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": result.names[class_id],
                    "confidence": round(float(box.conf.item()), 4),
                    "bbox": [round(float(value), 2) for value in box.xyxy[0].tolist()],
                }
            )

        return detections

    def predict(self, image, confidence=0.25):
        """Return structured detections for an image."""
        result = self._run_inference(image, confidence)
        return self._extract_detections(result)

    def predict_annotated(self, image, confidence=0.25):
        """Return an annotated image."""
        result = self._run_inference(image, confidence)
        return result.plot()

    def predict_with_annotation(self, image, confidence=0.25):
        """Return detections and annotated image from one inference."""
        result = self._run_inference(image, confidence)

        detections = self._extract_detections(result)
        annotated_image = result.plot()

        return detections, annotated_image
