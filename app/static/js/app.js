import { initializeUpload } from "./upload.js";
import { initializeCamera } from "./camera.js";

initializeUpload();
initializeCamera();

const confidenceSlider = document.getElementById("confidence-slider");
const confidenceValue = document.getElementById("confidence-value");
const thresholdStat = document.getElementById("threshold-stat");

function updateConfidenceDisplay() {
  const value = `${confidenceSlider.value}%`;

  confidenceValue.textContent = value;
  thresholdStat.textContent = value;
}

confidenceSlider.addEventListener("input", updateConfidenceDisplay);
updateConfidenceDisplay();
