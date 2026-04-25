# TrashAuto — Autonomous Garbage Collection Rover

An autonomous rover system that detects and collects garbage in a fixed indoor environment using a two-stage CNN inference pipeline. The system combines YOLOv5 for object detection and a fine-tuned ResNet-34 for binary garbage classification, running on-device on a Raspberry Pi.

**Real-world classification accuracy: 91.7%**

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [ML Pipeline](#ml-pipeline)
  - [Dataset](#dataset)
  - [Stage 1 — Object Detection (YOLOv5)](#stage-1--object-detection-yolov5)
  - [Stage 2 — Garbage Classification (ResNet-34)](#stage-2--garbage-classification-resnet-34)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Hardware](#hardware)
- [Sensor Integration & Navigation](#sensor-integration--navigation)
- [On-Device Inference Optimization](#on-device-inference-optimization)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [Future Work](#future-work)

---

## Overview

TrashAuto addresses the problem of automating garbage collection in a controlled indoor environment. Rather than relying on manual sorting or fixed conveyor-based systems, the rover autonomously navigates, identifies garbage objects in its field of view, classifies them, and performs collection actions.

The core challenge is reliable classification under real-world conditions — varying lighting, object orientations, partial occlusion, and background clutter — while running inference on a resource-constrained Raspberry Pi.

---

## System Architecture

```
Camera Input
     │
     ▼
┌─────────────┐
│   YOLOv5    │  ← Stage 1: Detect & localize all objects in frame
│  Detection  │
└──────┬──────┘
       │  Cropped object ROIs
       ▼
┌─────────────┐
│  ResNet-34  │  ← Stage 2: Classify each ROI as garbage / non-garbage
│ Classifier  │
└──────┬──────┘
       │  Classification result
       ▼
┌─────────────────────┐
│  Navigation &       │  ← Rover drives toward target, avoids obstacles,
│  Actuation Control  │     triggers collection mechanism
└─────────────────────┘
```

The two-stage design intentionally separates *where* (YOLO) from *what* (ResNet-34). This keeps the classification model lightweight and focused, rather than asking a single model to handle detection and fine-grained classification simultaneously.

---

## ML Pipeline

### Dataset

Training data was sourced from a combination of:
- **Public datasets:** TACO (Trash Annotations in Context) and a subset of OpenImages garbage-related categories
- **Custom collected data:** Images captured in the rover's actual operating environment to reduce domain gap between training distribution and deployment conditions

> **Note:** Custom data collection was critical. Models trained purely on public datasets showed significant accuracy degradation in the real environment due to lighting and background differences. Adding even a small number of in-environment images substantially improved real-world performance.

Data was split into train / validation / test sets. Both stages were trained and evaluated independently on their respective tasks.

---

### Stage 1 — Object Detection (YOLOv5)

| Detail | Value |
|---|---|
| Model | YOLOv5s (small variant) |
| Task | Object detection / localization |
| Input | Full camera frame |
| Output | Bounding boxes + confidence scores |
| Weights | Pretrained on COCO, fine-tuned on garbage dataset |

YOLOv5 runs first on every frame and produces bounding boxes around candidate objects. Each detected region of interest (ROI) is then cropped and passed to the ResNet-34 classifier. Detections below a confidence threshold are discarded before classification to reduce false positives fed into Stage 2.

---

### Stage 2 — Garbage Classification (ResNet-34)

| Detail | Value |
|---|---|
| Model | ResNet-34 |
| Task | Binary classification (garbage / non-garbage) |
| Input | Cropped ROI from YOLO detection |
| Output | Class label + confidence |
| Weights | Pretrained on ImageNet, fine-tuned on garbage dataset |

ResNet-34 was chosen over lighter alternatives (MobileNet, EfficientNet-B0) because accuracy on the classification task was prioritized over raw inference speed, given that YOLO's event-driven triggering already reduces how frequently the classifier runs.

Transfer learning from ImageNet weights significantly accelerated convergence and improved final accuracy compared to training from scratch, particularly given the limited size of the custom dataset component.

---

### Training

Both models were fine-tuned using PyTorch and fastai.

```bash
# Example: fine-tune ResNet-34 classifier
python train_classifier.py \
  --data_dir ./data/classification \
  --model resnet34 \
  --epochs 20 \
  --lr 1e-3 \
  --batch_size 32 \
  --pretrained
```

**Training configuration (ResNet-34):**
- Optimizer: Adam
- Learning rate schedule: One-cycle LR policy
- Loss: Cross-entropy
- Augmentation: Random horizontal flip, random rotation, color jitter, normalization

---

### Evaluation

The pipeline was evaluated end-to-end in the real operating environment, not just on held-out test data.

| Metric | Value |
|---|---|
| Real-world classification accuracy | **91.7%** |
| Evaluation environment | Fixed indoor environment |
| Evaluation method | Live rover runs across multiple test sessions |

Lab (test set) accuracy was higher than 91.7%. The gap between test set and real-world performance reflects domain shift from lighting variation and object placement differences — which motivated the addition of in-environment training data.

---

## Hardware

| Component | Purpose |
|---|---|
| Raspberry Pi 4 | Main compute — runs inference pipeline and navigation logic |
| Camera Module | Visual input for the ML pipeline |
| LiDAR sensor | Obstacle detection and distance measurement for navigation |
| Ultrasonic sensors | Short-range obstacle avoidance |
| IMU | Orientation tracking |
| Wheel encoders | Odometry for motion control |
| Drive motors + motor driver | Locomotion |
| Collection mechanism | Physical garbage pickup actuation |

---

## Sensor Integration & Navigation

Navigation uses a combination of LiDAR-based obstacle avoidance and camera-based target tracking:

- **LiDAR** provides 360° distance mapping for obstacle detection and path planning
- **Ultrasonic sensors** handle close-range stopping to avoid collisions
- **Wheel encoders** provide odometry feedback for dead-reckoning between waypoints
- **IMU** corrects for heading drift during navigation

Once a garbage object is classified, the rover uses its last known bounding box position to navigate toward the target and trigger the collection mechanism.

---

## On-Device Inference Optimization

Running two CNN models on a Raspberry Pi introduces latency and thermal constraints. The key optimization applied was **event-driven inference triggering**:

- The classification pipeline does not run on every camera frame
- YOLO detection only activates when motion or proximity sensors indicate an object is within range
- This significantly reduces average CPU load and prevents thermal throttling during extended operation

This approach trades a small amount of reaction latency for substantially lower power consumption and more consistent inference timing.

---

## Results

| Metric | Value |
|---|---|
| Real-world classification accuracy | 91.7% |
| Obstacle avoidance rate | — |
| Pipeline latency (end-to-end) | — |
| Operational environment | Fixed indoor |

> Fill in latency and obstacle avoidance numbers if available from test logs.

---

## Project Structure

```
trashauto/
├── data/
│   ├── detection/          # YOLO training data (images + labels)
│   └── classification/     # ResNet-34 training data (garbage / non-garbage)
├── models/
│   ├── yolo/               # YOLOv5 weights and config
│   └── classifier/         # ResNet-34 weights
├── src/
│   ├── detect.py           # YOLO inference wrapper
│   ├── classify.py         # ResNet-34 inference wrapper
│   ├── pipeline.py         # Two-stage pipeline orchestration
│   ├── navigation.py       # Rover navigation and sensor integration
│   └── control.py          # Motor control and actuation
├── train_detector.py       # YOLOv5 fine-tuning script
├── train_classifier.py     # ResNet-34 fine-tuning script
├── evaluate.py             # Evaluation script
├── requirements.txt
└── README.md
```

---

## Setup & Usage

### Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `torchvision`, `fastai`, `opencv-python`, `numpy`

### Run inference pipeline (on Raspberry Pi)

```bash
python src/pipeline.py --camera 0 --lidar /dev/ttyUSB0
```

### Train classifier

```bash
python train_classifier.py --data_dir ./data/classification --epochs 20
```

### Evaluate on test set

```bash
python evaluate.py --model_path ./models/classifier/resnet34.pth --data_dir ./data/classification/test
```

---

## Future Work

- [ ] Replace ResNet-34 with MobileNetV3 or EfficientNet-B0 to reduce inference latency on-device
- [ ] Add multi-class classification (plastic, metal, organic) rather than binary garbage / non-garbage
- [ ] Implement continuous learning loop to incorporate new in-environment samples over time
- [ ] Add quantization (INT8) to further reduce model size and inference cost on Raspberry Pi
- [ ] Extend to outdoor environments with larger and more diverse training data
