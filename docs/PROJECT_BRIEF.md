# Vision-Guided Recycling Sorter

**Implementation status:** The local end-to-end research application is implemented. Start with the [run instructions](#implemented-end-to-end-application) below. A trained synthetic demo is available; real-object accuracy and physical robot operation are not validated.

Vision-Guided Recycling Sorter
Vision-Guided Recycling Sorter is a production-oriented computer-vision application that helps a robot identify, locate, and sort recyclable objects placed on a table.

The system will receive an image from a fixed camera and produce a structured decision:
{
  "object_class": "plastic_bottle",
  "confidence": 0.92,
  "center_px": {"x": 418, "y": 267},
  "pick_point_mm": {"x": 134.2, "y": 86.7},
  "destination": "plastic_bin",
  "model_version": "resnet18-unet-v1"
}
The long-term application combines four capabilities:
1. Classification identifies the material or object type.
2. Semantic segmentation identifies the pixels belonging to each object.
3. Localization and calibration convert image positions into tabletop coordinates.
4. Robot integration turns the verified position into a simulated or physical pick command.
The project is designed to teach the complete path from a neural-network experiment to a reliable ML application. The first implementation uses a small CNN so that the forward pass, loss, backpropagation, and evaluation remain understandable. Later experiments compare it with a pretrained ResNet backbone and a U-Net or FCN segmentation model.
Product goal
Given a camera frame containing recyclable objects, return a confidence-aware, robot-ready sorting decision that another service or robot controller can consume through a stable interface.

The system must be able to say when it is uncertain. A production system should not issue a pick command when the object is outside the supported classes, the confidence is too low, the mask is poor, the camera calibration is invalid, or the image quality is insufficient.
Production-grade target
Production grade means the application will be built with the following requirements:
- Reproducible training and evaluation with versioned data splits, configurations, seeds, and model checkpoints.
- A clear separation between data loading, preprocessing, model inference, post-processing, calibration, and robot commands.
- Input validation for image size, color format, camera health, and missing or corrupted inputs.
- Versioned models with a model manifest containing training data, metrics, preprocessing, supported classes, and limitations.
- Confidence thresholds and an explicit unknown or needs_review outcome.
- Tests for preprocessing, tensor shapes, mask alignment, coordinate transforms, and API schemas.
- Evaluation on held-out data using classification accuracy, per-class precision and recall, segmentation IoU and Dice, localization error, false-pick rate, and inference latency.
- Structured logs containing request IDs, model versions, confidence, latency, and failure reasons without storing unnecessary personal data.
- A health check, readiness check, configuration through environment variables, and predictable error responses.
- A containerized inference service that can run locally and later be deployed to the cloud or an edge computer.
- Safe simulation before any physical robot is connected. The application will report a target position before it is permitted to command hardware.
Production quality does not mean claiming that the model is safe for unsupervised industrial use. The project will document its operating conditions, unsupported situations, and test coverage.
Planned architecture
Camera image
    ↓
Input validation
    ↓
Preprocessing
    ↓
Vision inference
    ├── object classification
    └── object segmentation
    ↓
Post-processing and confidence checks
    ↓
Pixel center and tabletop calibration
    ↓
Robot-ready pick decision
    ↓
Simulation or robot adapter
The model will not directly control a motor. A separate robot adapter will receive a validated, typed pick command. This keeps the computer-vision system testable and allows simulation before hardware integration.
Development phases
Phase 1: Foundation
- Set up the Python environment and project layout.
- Define supported classes and the image-label data contract.
- Load and normalize images.
- Run a small CNN forward pass.
- Add smoke tests and shape checks.
Phase 2: Classification
- Train the small CNN on a real dataset.
- Add train and validation splits.
- Track loss, accuracy, per-class metrics, and failure cases.
- Compare controlled changes such as augmentation and learning rate.
Phase 3: Transfer learning and explainability
- Use ResNet as a pretrained feature backbone.
- Compare frozen-backbone and fine-tuned models.
- Use Grad-CAM to inspect successful and failed predictions.
- Select a model using validation evidence.
Phase 4: Segmentation
- Prepare image-mask pairs.
- Train or adapt a U-Net or FCN model.
- Measure IoU and Dice.
- Reject masks that fail confidence or quality checks.
Phase 5: Localization and calibration
- Calculate object centers from predicted masks.
- Establish a tabletop coordinate system.
- Convert pixels to approximate millimeter coordinates.
- Measure localization error against known reference points.
Phase 6: Robot-ready application
- Expose inference through a stable local API or command.
- Produce typed JSON decisions.
- Add simulation for sorting actions.
- Add latency measurements and structured logs.
- Connect a physical robot only after simulation and safety checks pass.
Original foundation milestone
The repository initially contained only the foundation:
- The project layout
- The image and label data contract
- RGB image preprocessing
- A small four-class CNN
- A smoke test that verifies the image-to-model path
That foundation did not train, segment, track video, or control a robot. The implemented extension below adds paired-data training and segmentation; video tracking and physical control remain outside the implementation.
Setup
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
python scripts/smoke_test.py
Data contract
Each initial training example contains one image path and one integer class label. The initial classes are:
0 = can
1 = plastic_bottle
2 = paper
3 = other
Future stages will add pixel masks, object IDs, bounding geometry, camera metadata, and calibration information without changing the basic image-loading boundary.
Repository layout
- src/vision_sorter/data.py — class names and data definitions
- src/vision_sorter/preprocess.py — image conversion and tensor preparation
- src/vision_sorter/model.py — small CNN baseline
- scripts/smoke_test.py — end-to-end foundation check
- data/raw/ — downloaded source images, kept out of Git
- data/processed/ — generated datasets and splits, kept out of Git
Definition of done
The project is complete when a fresh environment can run one documented command that:
1. accepts a camera image,
2. validates and preprocesses it,
3. classifies supported objects,
4. produces segmentation masks and image coordinates,
5. rejects uncertain or unsupported cases,
6. returns a versioned structured decision,
7. records latency and failure information, and
8. demonstrates the resulting pick decision in simulation.
Learning target
At each phase, the implementation must be paired with an explanation of the relevant concept. The goal is to understand the complete path from pixels to a robot-ready decision, including where the system can fail and how those failures are measured.
