import { detectObjects } from "./api.js";

export function initializeCamera() {
  const cameraVideo = document.getElementById("camera-video");
  const startCameraButton = document.getElementById("start-camera");
  const stopCameraButton = document.getElementById("stop-camera");
  const cameraStatus = document.getElementById("camera-status");
  const cameraOverlay = document.getElementById("camera-overlay");
  const confidenceSlider = document.getElementById("confidence-slider");

  const overlayContext = cameraOverlay.getContext("2d");

  const captureCanvas = document.createElement("canvas");
  const captureContext = captureCanvas.getContext("2d");

  const LIVE_DETECTION_INTERVAL = 700;

  let cameraStream = null;
  let liveDetectionActive = false;
  let liveDetectionSession = 0;
  let liveDetectionAbort = null;
  let detectionTimer = null;
  let cameraRequestSession = 0;

  function clearOverlay() {
    overlayContext.clearRect(0, 0, cameraOverlay.width, cameraOverlay.height);
  }

  function drawCameraDetections(detections, imageWidth, imageHeight) {
    const width = cameraOverlay.width;
    const height = cameraOverlay.height;

    clearOverlay();

    const scaleX = width / imageWidth;
    const scaleY = height / imageHeight;

    detections.forEach((detection) => {
      const [x1, y1, x2, y2] = detection.bbox;

      const x = x1 * scaleX;
      const y = y1 * scaleY;
      const boxWidth = (x2 - x1) * scaleX;
      const boxHeight = (y2 - y1) * scaleY;

      const label = `${detection.class_name} ${(detection.confidence * 100).toFixed(1)}%`;

      overlayContext.strokeStyle = "#2dd4bf";
      overlayContext.lineWidth = 3;
      overlayContext.strokeRect(x, y, boxWidth, boxHeight);

      overlayContext.font = "bold 15px Arial";

      const labelWidth = overlayContext.measureText(label).width + 16;

      const labelY = Math.max(24, y);

      overlayContext.fillStyle = "#0f766e";
      overlayContext.fillRect(x, labelY - 24, labelWidth, 24);

      overlayContext.fillStyle = "#ffffff";
      overlayContext.fillText(label, x + 8, labelY - 7);
    });
  }

  function scheduleNextFrame(session) {
    if (!liveDetectionActive || session !== liveDetectionSession) {
      return;
    }

    detectionTimer = setTimeout(() => {
      detectionTimer = null;
      processCameraFrame(session);
    }, LIVE_DETECTION_INTERVAL);
  }

  async function processCameraFrame(session) {
    if (!liveDetectionActive || session !== liveDetectionSession) {
      return;
    }

    if (
      !cameraStream ||
      cameraVideo.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
      !cameraVideo.videoWidth
    ) {
      scheduleNextFrame(session);
      return;
    }

    try {
      const frameWidth = 640;
      const frameHeight = Math.round(
        (frameWidth * cameraVideo.videoHeight) / cameraVideo.videoWidth,
      );

      captureCanvas.width = frameWidth;
      captureCanvas.height = frameHeight;

      captureContext.drawImage(cameraVideo, 0, 0, frameWidth, frameHeight);

      const imageBlob = await new Promise((resolve) => {
        captureCanvas.toBlob(resolve, "image/jpeg", 0.8);
      });

      if (
        !imageBlob ||
        !liveDetectionActive ||
        session !== liveDetectionSession
      ) {
        return;
      }

      const controller = new AbortController();
      liveDetectionAbort = controller;

      const confidence = Number(confidenceSlider.value) / 100;

      const data = await detectObjects(
        imageBlob,
        confidence,
        controller.signal,
      );

      if (!liveDetectionActive || session !== liveDetectionSession) {
        return;
      }

      if (
        cameraOverlay.width !== cameraVideo.videoWidth ||
        cameraOverlay.height !== cameraVideo.videoHeight
      ) {
        cameraOverlay.width = cameraVideo.videoWidth;
        cameraOverlay.height = cameraVideo.videoHeight;
      }

      drawCameraDetections(
        data.detections,
        data.image_width,
        data.image_height,
      );

      cameraStatus.textContent = `Live detection running · ${data.detection_count} objects detected`;
    } catch (error) {
      if (error.name !== "AbortError") {
        console.error("Live detection error:", error);

        if (session === liveDetectionSession) {
          cameraStatus.textContent = error.message;
        }
      }
    } finally {
      if (session === liveDetectionSession) {
        liveDetectionAbort = null;
      }
      scheduleNextFrame(session);
    }
  }

  function startLiveDetection() {
    if (liveDetectionActive) return;

    liveDetectionActive = true;
    const session = ++liveDetectionSession;

    processCameraFrame(session);
  }

  function stopLiveDetection() {
    liveDetectionActive = false;
    ++liveDetectionSession;

    if (detectionTimer !== null) {
      clearTimeout(detectionTimer);
      detectionTimer = null;
    }

    if (liveDetectionAbort) {
      liveDetectionAbort.abort();
      liveDetectionAbort = null;
    }

    clearOverlay();
  }

  async function startCamera() {
    startCameraButton.disabled = true;
    cameraStatus.textContent = "Requesting camera access...";

    const requestSession = ++cameraRequestSession;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
        },
        audio: false,
      });

      if (requestSession !== cameraRequestSession) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      cameraStream = stream;
      cameraVideo.srcObject = stream;

      await cameraVideo.play();

      if (requestSession !== cameraRequestSession) {
        return;
      }

      stopCameraButton.disabled = false;
      cameraStatus.textContent = "Camera is running.";

      startLiveDetection();
    } catch (error) {
      if (requestSession !== cameraRequestSession) {
        return;
      }

      console.error("Camera error:", error);
      stopCamera();
      cameraStatus.textContent =
        "Unable to access camera. Check browser permissions.";
    }
  }

  function stopCamera() {
    ++cameraRequestSession;
    stopLiveDetection();

    if (cameraStream) {
      cameraStream.getTracks().forEach((track) => track.stop());
      cameraStream = null;
    }

    cameraVideo.pause();
    cameraVideo.srcObject = null;

    startCameraButton.disabled = false;
    stopCameraButton.disabled = true;
    cameraStatus.textContent = "Camera stopped.";
  }

  startCameraButton.addEventListener("click", startCamera);
  stopCameraButton.addEventListener("click", stopCamera);
  window.addEventListener("pagehide", stopCamera);
}
