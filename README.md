
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

Model evaluation determines how accurately and efficiently a trained model performs.

### Evaluation Metrics

| Metric | Explanation | Importance |
|---|---|---|
| Precision | Percentage of predicted objects that are correct | Helps measure false alarms |
| Recall | Percentage of actual objects successfully detected | Helps measure missed objects |
| mAP@50 | Detection performance using a 50% bounding-box overlap threshold | Measures detection quality |
| mAP@50–95 | Average detection performance across several overlap thresholds | Provides a stricter evaluation |
| Per-Class Performance | Detection results for individual object categories | Helps identify weaker classes |
| Inference Time | Time required to generate a prediction | Measures model speed |
| Model Size | Storage required by the model | Helps assess deployment requirements |

### Understanding Prediction Outcomes

**True Positive (TP)**

The model correctly identifies an object.

Example: A shock absorber is present and the model detects it.

**False Positive (FP)**

The model predicts an object that is not actually present.

Example: A person's face is incorrectly classified as a cylinder.

**False Negative (FN)**

The model misses an object that is actually present.

Example: A cylinder is visible, but the model does not detect it.

### Understanding IoU

Intersection over Union (IoU) measures how closely a predicted bounding box overlaps the actual labelled bounding box.

A higher IoU generally indicates more accurate object localisation.

### Model Inference Performance

The selected YOLO26n model was benchmarked locally using a synthetic 640 × 640 image.

The benchmark included three warm-up predictions followed by ten measured predictions.

| Metric | Result |
|---|---:|
| Average inference time | 22.8 ms |
| Minimum inference time | 22.3 ms |
| Maximum inference time | 23.3 ms |

These results measure model inference under local test conditions.

### API Performance

The combined detection API was tested using 20 sequential requests after three warm-up requests.

| Metric | Result |
|---|---:|
| Average response time | 27.8 ms |
| Median response time | 26.9 ms |
| Minimum response time | 25.8 ms |
| Maximum response time | 43.6 ms |
| P95 response time | 29.4 ms |

The benchmark used a synthetic PNG image and the `/api/detect/combined` endpoint.

### Performance Interpretation

The results demonstrate that the application can provide responsive object detection in the tested local environment.

However:

- These measurements are not production performance guarantees.
- Results may differ depending on hardware.
- API latency does not directly represent live camera frame rate.
- Fast inference does not necessarily mean high detection accuracy.

### Evaluation Conclusion

The selected model successfully detected real shock absorbers during manual testing.

The model also produced false-positive detections, showing that further testing with varied backgrounds and unrelated objects would be valuable.

The project's local speed benchmarks and functional tests demonstrate application performance, while a full assessment of detection accuracy requires the labelled test-set evaluation results.

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
