# Video ADAS Risk Alerts on BDD100K  
**YOLOv8 + DeepLabV3 + Tracking + Rule-based Reasoning**

---

## 1. Project Objective

This project builds a **production-like ADAS pipeline** that operates directly on **raw traffic videos** (no labels required at inference time).  
The system outputs:

- **Overlay video**: bounding boxes, track IDs, semantic context (road/sidewalk), and risk alerts  
- **Event log** (JSONL / CSV): structured records of detected risk events over time

The pipeline combines:
- **YOLOv8 (object-level perception)** for fast and robust detection
- **DeepLabV3 (scene-level semantics)** to understand road context
- **Multi-object tracking** for temporal consistency
- **Rule-based reasoning** to infer interpretable safety-related events

---

## 2. Dataset

### 2.1 BDD100K Overview
BDD100K is a large-scale autonomous driving dataset containing images and videos captured under diverse conditions:
- day / night
- various weather
- urban and highway scenes

It provides annotations for multiple tasks, including detection and semantic segmentation.

---

### 2.2 Detection Dataset (YOLO)

- **Task**: Object Detection  
- **Classes (5)**:
  - `car`
  - `person`
  - `truck`
  - `bus`
  - `traffic_control`
- **Split**:
  - Train: ~8,000 images  
  - Validation: ~1,000 images  
  - Test: ~1,000 images  

**Note**:  
`traffic_control` is a particularly challenging class due to:
- small object size
- frequent occlusion
- motion blur and night scenes

---

### 2.3 Segmentation Dataset (DeepLab)

- **Task**: Semantic Segmentation  
- Original BDD100K masks are **remapped to 6 classes**:
  - `0` – other  
  - `1` – road  
  - `2` – sidewalk  
  - `3` – person  
  - `4` – rider  
  - `5` – vehicle  
  - `255` – ignore  

- **Split**:
  - Train: 7,000  
  - Validation: 1,000  
  - Test: 500  

The test set is sampled from training data to guarantee valid image–mask correspondence.

---

## 3. Models

### 3.1 YOLOv8s — Object Detection

- Architecture: YOLOv8s (Ultralytics)
- Input: RGB image
- Output: bounding boxes, class labels, confidence scores

**Role in the pipeline**:
- Detects concrete traffic participants (vehicles, pedestrians)
- Supplies bounding boxes for tracking
- Acts as the primary signal for object-level reasoning

---

### 3.2 DeepLabV3-ResNet50 — Semantic Segmentation

- Architecture: DeepLabV3 with ResNet50 backbone (torchvision)
- Input: RGB image
- Output: pixel-level semantic map

**Role in the pipeline**:
- Provides **scene context** (road vs. sidewalk)
- Does **not** replace detection
- Enables spatial reasoning such as:
  - “Is a pedestrian standing on the road?”
  - “Is a vehicle driving onto the sidewalk?”

---

## 4. Training Results

### 4.1 YOLOv8s (5 classes) — Test Set

**Checkpoint**: `bd100k_5cls_8k_yv8s_ft_768/weights/best.pt`  
**Input size**: 768 (fine-tuned)

**Overall metrics (TEST)**:
- Precision: **0.705**
- Recall: **0.558**
- mAP@50: **0.626**
- mAP@50–95: **0.381**

**Per-class performance (TEST)**:
- `car`: mAP@50 **0.779** | mAP@50–95 **0.487**
- `bus`: **0.600** | **0.455**
- `truck`: **0.577** | **0.410**
- `person`: **0.575** | **0.279**
- `traffic_control`: **0.598** | **0.274**

---

#### Why do some YOLO metrics look weak?

**(A) Recall < Precision (0.558 vs 0.705)**  
The model is conservative:
- detections are reliable when present
- but many objects are missed

Common causes:
- small or distant objects
- occlusion
- night scenes and motion blur
- relatively strict confidence/NMS thresholds

---

**(B) Large gap between mAP@50 and mAP@50–95**  
- mAP@50 only requires IoU ≥ 0.5  
- mAP@50–95 evaluates multiple stricter IoU thresholds  

Classes such as `person` and `traffic_control`:
- are small and thin
- have ambiguous bounding box boundaries
- suffer from label noise  

→ bounding boxes are correct but not tight enough, reducing mAP@50–95.

---

**(C) `person` performs worse than `car`**  
Pedestrians:
- occupy fewer pixels
- exhibit large pose variation
- are often partially occluded  

Detection and precise localization are therefore harder.

---

**(D) Stable generalization**  
- Validation mAP@50–95: **0.372**
- Test mAP@50–95: **0.381**

The small gap (+0.009) indicates **no clear overfitting**.

---

### 4.2 DeepLabV3-ResNet50 (6 classes) — Test Set

- Best epoch: 18  
- Best validation mIoU: **0.7015**
- Test mIoU: **0.6899**
- Test loss: **0.3349**

**Per-class IoU (TEST)**:
- `other`: **0.9275**
- `road`: **0.5579**
- `sidewalk`: **0.7956**
- `person`: **0.3333**
- `rider`: **0.8523**
- `vehicle`: **0.6729**

---

#### Why do some DeepLab IoUs look low?

**(A) Low IoU for `person` (~0.33)**  
Pixel-level person segmentation is inherently difficult:
- thin structures (arms, legs)
- occlusions
- small pixel footprint
- severe class imbalance  

This behavior is expected in street-scene segmentation.

---

**(B) Road IoU lower than sidewalk IoU**  
Road regions are visually complex:
- shadows and reflections (especially at night)
- lane markings and glare
- ambiguous road–sidewalk boundaries  

This causes confusion between `road`, `sidewalk`, and `other`.

---

**(C) Very high IoU for `other`**  
- `other` covers large background areas
- large-area classes are easier to segment
- but may mask errors in smaller classes due to imbalance

---

**Pipeline relevance**  
The system **does not rely on `person` segmentation**.  
DeepLab is primarily used for **road and sidewalk**, where performance is sufficient for intersection-based rules.

---

## 5. Video ADAS Risk-Alert Pipeline

### 5.1 Input / Output

**Input**:  
- Raw traffic video (no annotations required)

**Output**:
- Overlay video with detections, tracking IDs, semantic context, and alerts. Link: https://drive.google.com/drive/folders/1oskZTnyBzhOaHizTjpxL7t8G7J1Zp5Cn?usp=sharing
- Event log (JSONL/CSV), e.g.:

```json
{
  "t_sec": 12.53,
  "frame_idx": 376,
  "track_id": 19,
  "cls": "person",
  "event_type": "PedestrianOnRoad",
  "score": 0.81,
  "bbox_xyxy": [x1, y1, x2, y2],
  "extras": { "ratio_road": 0.34 }
}
