from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.detection import router as detection_router

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Hazardous Waste Object Detection API",
    description="YOLO-based detection of Shock Absorbers and cylinders",
    version="1.0.0",
)

# Serve CSS, JavaScript and other static assets.
app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)

# Register detection API endpoints.
app.include_router(detection_router)


@app.get("/", include_in_schema=False)
def home():
    """Serve the HazWaste Vision dashboard."""
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "hazwaste-yolo-api",
    }
