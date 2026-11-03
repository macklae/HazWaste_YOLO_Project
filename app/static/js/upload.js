import { detectCombined } from "./api.js";

export function initializeUpload() {
  const imageInput = document.getElementById("image-input");
  const uploadZone = document.getElementById("upload-zone");
  const selectedFileLabel = document.getElementById("selected-file");
  const analyzeButton = document.getElementById("analyze-button");

  const confidenceSlider = document.getElementById("confidence-slider");

  const resultImage = document.getElementById("result-image");
  const emptyPreview = document.getElementById("empty-preview");
  const resultStatus = document.getElementById("result-status");

  const totalDetections = document.getElementById("total-detections");
  const shockCount = document.getElementById("shock-count");
  const cylinderCount = document.getElementById("cylinder-count");
  const detectionsTable = document.getElementById("detections-table");

  const MAX_IMAGE_SIZE = 10 * 1024 * 1024;

  let selectedFile = null;
  let currentImageUrl = null;
  let analyzing = false;

  // =====================================
  // Status and result management
  // =====================================

  function setStatus(message) {
    resultStatus.textContent = message;
  }

  function resetResults() {
    totalDetections.textContent = "0";
    shockCount.textContent = "0";
    cylinderCount.textContent = "0";

    detectionsTable.replaceChildren();

    const row = detectionsTable.insertRow();
    const cell = row.insertCell();

    cell.colSpan = 4;
    cell.className = "empty-table";
    cell.textContent = "No detections to display";

    resultImage.hidden = true;
    emptyPreview.hidden = false;

    if (currentImageUrl) {
      URL.revokeObjectURL(currentImageUrl);
      currentImageUrl = null;
    }

    resultImage.removeAttribute("src");
  }

  // =====================================
  // Image selection and validation
  // =====================================

  function selectImage(file) {
    if (analyzing || !file) return;

    const allowedTypes = ["image/jpeg", "image/png"];

    if (!allowedTypes.includes(file.type)) {
      setStatus("Please select a JPEG or PNG image.");
      return;
    }

    if (file.size > MAX_IMAGE_SIZE) {
      setStatus("Image exceeds the 10 MB limit.");
      return;
    }

    selectedFile = file;
    selectedFileLabel.textContent = file.name;
    analyzeButton.disabled = false;

    resetResults();
    setStatus("Image ready for analysis.");
  }

  // =====================================
  // Detection statistics
  // =====================================

  function updateStatistics(detections) {
    totalDetections.textContent = detections.length;

    shockCount.textContent = detections.filter(
      (item) => item.class_id === 0,
    ).length;

    cylinderCount.textContent = detections.filter(
      (item) => item.class_id === 1,
    ).length;
  }

  function updateDetectionTable(detections) {
    detectionsTable.replaceChildren();

    if (detections.length === 0) {
      const row = detectionsTable.insertRow();
      const cell = row.insertCell();

      cell.colSpan = 4;
      cell.className = "empty-table";
      cell.textContent = "No objects detected.";

      return;
    }

    detections.forEach((detection, index) => {
      const row = detectionsTable.insertRow();

      row.insertCell().textContent = index + 1;
      row.insertCell().textContent = detection.class_name;

      row.insertCell().textContent = `${(detection.confidence * 100).toFixed(2)}%`;

      row.insertCell().textContent = detection.bbox.join(", ");
    });
  }

  // =====================================
  // YOLO image detection
  // =====================================

  async function runDetection() {
    if (!selectedFile || analyzing) return;

    analyzing = true;
    analyzeButton.disabled = true;
    analyzeButton.textContent = "Analyzing...";

    setStatus("Running YOLO26n object detection...");

    const file = selectedFile;
    const confidence = Number(confidenceSlider.value) / 100;

    try {
      // One API request, one YOLO inference.
      const detectionData = await detectCombined(file, confidence);

      // Display the annotated image returned by the combined API.
      resultImage.src = detectionData.annotated_image;
      resultImage.hidden = false;
      emptyPreview.hidden = true;

      // Display detection statistics and bounding boxes.
      updateStatistics(detectionData.detections);
      updateDetectionTable(detectionData.detections);

      setStatus(
        `Analysis complete. ${detectionData.detection_count} objects detected.`,
      );
    } catch (error) {
      console.error("Upload detection error:", error);
      resetResults();
      setStatus(error.message);
    } finally {
      analyzing = false;
      analyzeButton.disabled = false;
      analyzeButton.textContent = "Run Detection";
    }
  }

  // =====================================
  // Event listeners
  // =====================================

  imageInput.addEventListener("change", () => {
    selectImage(imageInput.files[0]);
  });

  uploadZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    uploadZone.classList.add("drag-over");
  });

  uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("drag-over");
  });

  uploadZone.addEventListener("drop", (event) => {
    event.preventDefault();
    uploadZone.classList.remove("drag-over");

    selectImage(event.dataTransfer.files[0]);
  });

  analyzeButton.addEventListener("click", runDetection);

  // Release image resources when leaving the page.
  window.addEventListener("pagehide", () => {
    if (currentImageUrl) {
      URL.revokeObjectURL(currentImageUrl);
      currentImageUrl = null;
    }
  });
}
