
# HazWaste Vision — Hazardous Object Detection Using YOLO

HazWaste Vision is a computer vision application that detects **shock absorbers** and **cylinders** in uploaded images and live camera feeds.

The project demonstrates the end-to-end machine learning lifecycle, from understanding a business problem and preparing data to building, testing, evaluating and deploying an object detection model.

The application uses **YOLO26n**, **FastAPI**, **Python**, and a web-based user interface.

## 1. Problem Statement & Business Understanding

### Business Problem

Identifying hazardous or potentially hazardous objects during visual inspection can require significant manual effort.

Manual inspection can be:
- Time-consuming.
- Inconsistent between inspectors.
- Difficult to scale when processing large numbers of images.
- Prone to missed objects.

### Proposed Solution

Develop an AI-powered object detection application that can:

- Identify shock absorbers and cylinders in images.
- Draw bounding boxes around detected objects.
- Display object names and confidence scores.
- Detect objects using a live camera.
- Allow users to adjust detection confidence thresholds.
- Provide APIs for integration with other systems.

### Business Value

The application demonstrates how computer vision can support inspection workflows by:

- Reducing repetitive manual screening.
- Helping users locate objects of interest.
- Providing consistent detection outputs.
- Supporting faster review of images.

**Important:** This is a proof-of-concept application, not a certified workplace safety system. Human verification remains necessary.

### Project Success Criteria

The project aims to:

1. Detect the two target object classes.
2. Minimise incorrect detections and missed objects.
3. Provide responsive image and camera detection.
4. Offer a simple web interface.
5. Handle invalid inputs reliably.
6. Provide automated tests and reproducible setup instructions.

## 2. Data Collection

The model was developed to detect two object categories:

| Class ID | Object |
|---|---|
| 0 | Shock Absorber |
| 1 | Cylinder |

The project uses labelled images for object detection.

Each object annotation identifies:

- The object class.
- The object's location in the image.
- The area covered by its bounding box.

### Dataset Organisation

Object detection datasets are typically organised into three groups:

| Dataset | Purpose |
|---|---|
| Training | Used to teach the model to recognise objects |
| Validation | Used to monitor training and compare models |
| Testing | Used to evaluate performance on unseen images |

The project's training notebook provides the training workflow. Exact dataset counts and source-version details should be taken from the original training records rather than estimated.

## 3. Data Understanding

Data understanding involves examining the images and their annotations before training.

Important considerations include:

- The number of images available.
- The number of labelled objects in each class.
- Whether one class has more examples than another.
- Variations in lighting and camera angles.
- Differences in object sizes and backgrounds.
- Whether objects are partially hidden.
- Whether bounding boxes correctly identify objects.

### Why Class Distribution Matters

If a dataset contains significantly more examples of one class, a model may perform better on that class.

For example, if cylinders appear much more frequently than shock absorbers, the model may have fewer opportunities to learn the different appearances of shock absorbers.

### Real-World Data Challenges

Images captured outside a controlled training environment may contain:

- Unfamiliar backgrounds.
- Different lighting conditions.
- Unexpected camera angles.
- Objects that resemble the target classes.
- Multiple objects within the same frame.

These differences can affect detection accuracy.

## 4. Data Preparation / Cleansing

Data preparation helps ensure that the images and annotations are suitable for training.

### Key Preparation Activities

1. Organise images and labels into appropriate folders.
2. Maintain consistent class IDs.
3. Check image readability and annotation formats.
4. Prepare images for YOLO processing.
5. Review class distribution.
6. Consider augmentation to improve training-image variety.

### Image Processing

YOLO processes images at a configured input size.

The deployed application uses an inference image size of **640 pixels**.

Image preprocessing is handled by the model's inference pipeline.

### Data Augmentation

Data augmentation creates variations of existing training images to help a model learn from different visual conditions.

Examples include:

- Flipping images.
- Adjusting brightness.
- Changing image orientation.
- Applying small visual transformations.

Augmentation must be applied carefully because unrealistic transformations can reduce model performance.

### Data Preparation Assumptions

The model development approach assumes that:

- Labels correctly identify the objects.
- Class IDs remain consistent.
- Training and testing images are independent.
- The test dataset has not been used to train the model.
- Images reasonably represent the objects the application needs to detect.

## 5. Model Building

### Why YOLO?

YOLO stands for **You Only Look Once**.

It is a deep learning architecture designed for object detection.

Unlike image classification, which predicts the main category of an image, object detection identifies both:

1. **What object is present.**
2. **Where the object is located.**

YOLO can also detect multiple objects in a single image.

### How the Model Works

```text
Input Image
     |
     v
YOLO Model
     |
     v
Object Detection
     |
     v
Bounding Boxes
     |
     v
Class Labels + Confidence Scores
```

### Selected Model

The deployed application uses a custom-trained **YOLO26n** model.

Model file:

`models/yolo26n_original_best.pt`

Approximate model size: **5.2 MB**.

The model has been trained to recognise:

- Shock Absorber
- Cylinder

### Why a Compact Model?

A smaller object detection model can offer practical advantages:

- Lower storage requirements.
- Faster inference.
- Reduced computing requirements.
- Easier integration into lightweight applications.

Model size alone does not establish detection accuracy. Accuracy must be evaluated separately using labelled test images.

## 6. Model Training

### Training Approach

The YOLO model was trained using labelled images of shock absorbers and cylinders.

During training:

1. Images are passed into the model.
2. The model predicts object classes and bounding boxes.
3. Predictions are compared with the labelled objects.
4. The training process calculates errors.
5. Model parameters are updated to reduce those errors.
6. Validation results are used to monitor performance.
7. The best-performing model checkpoint is saved.

### Important Training Parameters

| Parameter | Meaning |
|---|---|
| Epochs | Number of times the model can process the training dataset |
| Batch Size | Number of images processed together |
| Image Size | Resolution used during model processing |
| Learning Rate | Controls how much the model changes during training |
| Patience | Controls early stopping when performance stops improving |
| Model Weights | Learned parameters saved after training |

The repository includes the training notebook:

`Hazardous_Object_Detection_Using_YOLO.ipynb`

The deployed model checkpoint is:

`models/yolo26n_original_best.pt`

### Training Assumptions

- The dataset contains correctly labelled examples.
- The selected model checkpoint represents the intended training run.
- Validation performance is used to assess training progress.
- The model may perform differently on images outside the training distribution.

## 7. Model Testing

Testing was performed at both the model and application levels.

### Functional Testing

The following application features were tested:

- Image uploads.
- Object detection.
- Bounding-box display.
- Class labels and confidence scores.
- Live camera detection.
- Camera start and stop behaviour.
- Confidence threshold adjustment.
- Detection API responses.
- Invalid image handling.
- Oversized image handling.

### Automated Testing

The repository contains two automated test files:

```text
tests/
├── test_api.py
└── test_hardening.py
```

The tests cover API functionality, input validation, detection responses and application reliability.

### Automated Test Results

| Metric | Result |
|---|---:|
| Tests executed | 18 |
| Tests passed | 18 |
| Tests failed | 0 |

**Result: All 18 automated tests passed.**

A non-blocking dependency deprecation warning was reported during testing.

### Run Automated Tests

```bash
uv run python -m pytest -q
```

### Code Quality Checks

```bash
uv run ruff check app tests
uv run ruff format --check app tests
```

### Manual Testing Results

A real shock absorber was successfully detected using the live camera.

However, testing also revealed a limitation: a person's face was incorrectly classified as a cylinder in some camera frames, including a prediction with approximately **82% confidence**.

One image upload also returned HTTP 400 because the uploaded image content did not match its declared format.

These observations are documented as limitations rather than treated as resolved issues.


## 8. Model Evaluation

The objective was to identify a model that balances detection accuracy, missed detections, false alarms, model size and processing speed.

### Evaluation Dataset

The original model comparison used a held-out test dataset containing:

- **14 test images**
- **49 labelled objects**
- **15 shock absorbers**
- **34 cylinders**

The test dataset was used to measure how well the trained models performed on images outside their training data.

Because the test dataset was relatively small, these results should be treated as initial evaluation findings rather than a guarantee of real-world accuracy.

### Evaluation Metrics

| Metric | What it measures |
|---|---|
| Precision | How many predicted objects were correct |
| Recall | How many actual objects were successfully detected |
| mAP@50 | Overall detection performance at a 50% bounding-box overlap threshold |
| mAP@50–95 | Average detection performance across increasingly strict overlap thresholds |
| Per-class Recall | Detection success rate for each object category |
| Parameters | Number of learnable parameters in the model |
| GFLOPs | Approximate computational work required for inference |
| Inference Time | Time required to process an image |

### Model Comparison — Original Training

| Model | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|
| YOLOv8n | 0.830 | 0.435 | **0.556** | 0.333 |
| YOLOv8s | **0.848** | **0.453** | 0.503 | 0.326 |
| YOLO26n | 0.676 | 0.438 | 0.535 | **0.378** |

**Key findings:**

- YOLOv8s achieved the highest precision and overall recall.
- YOLOv8n achieved the highest mAP@50.
- YOLO26n achieved the highest mAP@50–95.
- No single model performed best across every metric.

### Per-Class Recall — Original Training

| Model | Shock Absorber Recall | Cylinder Recall |
|---|---:|---:|
| YOLOv8n | 0.400 | 0.471 |
| YOLOv8s | 0.400 | 0.506 |
| YOLO26n | 0.333 | **0.543** |

The selected YOLO26n model detected approximately 33% of labelled shock absorbers and 54% of labelled cylinders in the test set, according to the recorded per-class recall values.

These results show that missed detections remain an important limitation, especially for shock absorbers.

### Original vs Balanced Training

Additional experiments investigated whether balancing and augmenting the training data could improve detection.

| Model | Training | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---|---:|---:|---:|---:|
| YOLOv8n | Original | 0.830 | 0.435 | 0.556 | 0.333 |
| YOLOv8n | Balanced | 0.631 | 0.513 | 0.512 | 0.313 |
| YOLOv8s | Original | 0.848 | 0.453 | 0.503 | 0.326 |
| YOLOv8s | Balanced | 0.526 | 0.509 | 0.493 | 0.301 |
| YOLO26n | Original | 0.676 | 0.438 | 0.535 | 0.378 |
| YOLO26n | Balanced | 0.490 | 0.447 | 0.452 | 0.297 |

**Findings:**

- Balancing improved overall recall for all three models.
- Precision decreased for all three models.
- mAP@50–95 also decreased for all three models.
- The experiments demonstrated a trade-off between detecting more objects and maintaining prediction quality.

The balanced results are the values recorded in the experiment comparison summary. Differences previously noted between summary entries and saved test outputs should be reconciled before treating these as independently verified results.

### Model Complexity Comparison

| Model | Parameters | GFLOPs | Recorded Inference Time |
|---|---:|---:|---:|
| YOLOv8n | 3.006M | 8.1 | ~6.8 ms |
| YOLOv8s | 11.126M | 28.4 | ~12.5 ms |
| YOLO26n | **2.375M** | **5.3** | **~5.3 ms** |

YOLO26n had the smallest model footprint and lowest recorded computational requirement among the three original models.

### Final Model Selection

**Selected model: YOLO26n — Original Training**

The model was selected because:

1. It achieved the highest mAP@50–95 of **0.378**.
2. It had the smallest parameter count at **2.375 million**.
3. It required the lowest recorded computational workload at **5.3 GFLOPs**.
4. It provided a practical balance between model size, processing speed and detection performance.

However, its relatively low recall means it is not yet suitable for safety-critical decisions without further improvements and human verification.

### Local Model Inference Benchmark

The deployed YOLO26n model was separately benchmarked using a synthetic 640 × 640 image, with three warm-up predictions and ten measured predictions.

| Metric | Result |
|---|---:|
| Average inference time | **22.8 ms** |
| Minimum inference time | 22.3 ms |
| Maximum inference time | 23.3 ms |

These local application measurements differ from the earlier model-comparison inference timings because the benchmark conditions were different. They should not be compared directly.

### API Performance Benchmark

The combined detection API was benchmarked using 20 sequential requests after three warm-up requests.

| Metric | Result |
|---|---:|
| Average response time | **27.8 ms** |
| Median response time | 26.9 ms |
| Minimum response time | 25.8 ms |
| Maximum response time | 43.6 ms |
| P95 response time | 29.4 ms |

The benchmark used a synthetic PNG image and the `/api/detect/combined` endpoint.

These measurements represent local test conditions, not guaranteed production response times.

### Application Quality Evaluation

| Measurement | Result |
|---|---|
| Automated tests | **18 passed, 0 failed** |
| Live camera detection | Successfully detected a real shock absorber |
| False-positive observation | A face was incorrectly detected as a cylinder at approximately 82% confidence |
| Input validation | JPEG/PNG checks, 10 MB upload limit and 20-megapixel resolution limit |
| Concurrency protection | Model inference protected by a lock |

### Overall Evaluation Conclusion

The experiments demonstrated that YOLO26n offered the strongest mAP@50–95 result among the original models while maintaining the smallest computational footprint.

Balanced training improved recall but reduced precision and mAP@50–95, so the original YOLO26n checkpoint was retained for deployment.

The deployed application passed all 18 automated tests and achieved an average local API response time of 27.8 ms.

**Important limitation:** Detection recall remained relatively low, particularly for shock absorbers. More training data, improved class representation, additional negative examples and testing on a larger independent dataset are recommended before any safety-sensitive use.


## 9. Model Deployment

### Application Architecture

```text
User
  |
  v
Web Browser
(HTML, CSS, JavaScript)
  |
  |--- Image Upload
  |
  |--- Live Camera
  |
  v
FastAPI Backend
  |
  v
Image Validation
  |
  v
YOLO26n Model
  |
  v
Object Detection
  |
  v
Bounding Boxes + Confidence Scores
  |
  v
Results Displayed in Browser
```

### Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Programming Language | Python | Backend development |
| Object Detection | Ultralytics YOLO26n | Detect target objects |
| API Framework | FastAPI | Expose detection services |
| Web Server | Uvicorn | Run the backend |
| Image Processing | Pillow and OpenCV | Process images |
| Frontend | HTML, CSS, JavaScript | Web interface |
| Environment Management | uv | Manage dependencies |
| Testing | pytest | Automated testing |
| Code Quality | Ruff | Linting and formatting |

### Project Structure

```text
HazWaste_YOLO_Project/
├── app/
│   ├── api/
│   │   └── detection.py
│   ├── services/
│   │   └── inference.py
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css
│   │   └── js/
│   │       ├── api.js
│   │       ├── app.js
│   │       ├── camera.js
│   │       └── upload.js
│   ├── templates/
│   │   └── index.html
│   └── main.py
├── models/
│   └── yolo26n_original_best.pt
├── tests/
│   ├── test_api.py
│   └── test_hardening.py
├── Hazardous_Object_Detection_Using_YOLO.ipynb
├── benchmark_api.py
├── pyproject.toml
├── uv.lock
└── README.md
```

### Running the Application Locally

The project was developed using Python 3.11 and uv.

**Step 1 — Clone the repository**

```bash
git clone https://github.com/macklae/HazWaste_YOLO_Project.git
cd HazWaste_YOLO_Project
```

**Step 2 — Install dependencies**

```bash
uv sync
```

**Step 3 — Start the application**

```bash
uv run uvicorn app.main:app --reload
```

**Step 4 — Open the application**

Open:

http://127.0.0.1:8000/

The `--reload` option is intended for local development.

### API Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /` | Open the web application |
| `GET /api/health` | Check API health |
| `POST /api/detect` | Return object detection results |
| `POST /api/detect/annotated` | Return an image containing bounding boxes |
| `POST /api/detect/combined` | Return detection details and an annotated image |

Interactive API documentation:

http://127.0.0.1:8000/docs

### Input Validation and Reliability

The backend includes:

- JPEG and PNG format validation.
- Empty and corrupted file handling.
- Maximum upload size of 10 MB.
- Maximum image resolution of 20 megapixels.
- Confidence threshold validation.
- A lock around model inference to prevent simultaneous access to the shared model.

### Deployment Status

The application has been successfully tested in a local development environment.

The source code and selected trained model are maintained in the GitHub repository.

GitHub hosts the project files but does not automatically run the application as a public web service.

## Assumptions, Limitations & Future Improvements

### Assumptions

- Dataset annotations are accurate.
- Training and test datasets are independent.
- The selected model checkpoint corresponds to the intended training experiment.
- Local performance results may differ on other machines.
- Human verification remains necessary for safety-related decisions.

### Known Limitations

- False-positive cylinder detections can occur in live camera frames.
- A high confidence score does not guarantee a correct prediction.
- One JPG upload returned a format mismatch error.
- Production-scale load and security testing have not been completed.
- Detection accuracy across diverse real-world environments requires further evaluation.

### Future Improvements

1. Expand the dataset with additional real-world images.
2. Add negative examples containing unrelated objects.
3. Evaluate performance on a larger independent test set.
4. Compare retrained models against the current checkpoint.
5. Investigate reproducible valid-JPEG upload failures.
6. Add automated GitHub Actions testing.
7. Consider cloud deployment after further reliability testing.

## Reproducibility & Responsible Use

The repository includes:

- Application source code.
- The selected trained YOLO model.
- The training notebook.
- Automated tests.
- Dependency definitions.
- Local benchmarking code.

Reproducing the training results requires the original dataset, annotations and training configuration.

This project is intended for learning, demonstration and portfolio purposes.

It is **not a certified safety system** and should not be used as the sole basis for workplace safety decisions.

---

**Author:** [macklae](https://github.com/macklae)

**Project:** HazWaste Vision

**Status:** Local application tested and validated; known limitations documented.
