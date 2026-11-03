const API_BASE = "/api";

// Used by live camera detection.
export async function detectObjects(file, confidence, signal = undefined) {
  const formData = new FormData();

  formData.append("file", file);
  formData.append("confidence", confidence.toString());

  const response = await fetch(`${API_BASE}/detect`, {
    method: "POST",
    body: formData,
    signal,
  });

  if (!response.ok) {
    throw new Error(`Object detection failed: HTTP ${response.status}`);
  }

  return response.json();
}

// Existing annotated-image API.
export async function getAnnotatedImage(file, confidence, signal = undefined) {
  const formData = new FormData();

  formData.append("file", file);
  formData.append("confidence", confidence.toString());

  const response = await fetch(`${API_BASE}/detect/annotated`, {
    method: "POST",
    body: formData,
    signal,
  });

  if (!response.ok) {
    throw new Error(`Annotated image failed: HTTP ${response.status}`);
  }

  return response.blob();
}

// Combined API: one request and one YOLO inference.
export async function detectCombined(file, confidence, signal = undefined) {
  const formData = new FormData();

  formData.append("file", file);
  formData.append("confidence", confidence.toString());

  const response = await fetch(`${API_BASE}/detect/combined`, {
    method: "POST",
    body: formData,
    signal,
  });

  if (!response.ok) {
    throw new Error(`Combined detection failed: HTTP ${response.status}`);
  }

  return response.json();
}
