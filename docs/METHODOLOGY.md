# Methodology

This document summarizes the implementation-level methodology used by TopoEKF.

## 1. Detection

Input frames are processed with a YOLO detector. Each detection is represented as:

```text
d_i = (bbox_i, centroid_i, confidence_i, class_id_i)
```

For UAV vehicle tracking, the default class filter targets common vehicle classes and uses a low confidence threshold to preserve small aerial objects.

## 2. Data Association

Predicted track positions and detection centroids are matched with a Mahalanobis-distance cost matrix:

```text
d_M(i,j) = sqrt((z_j - H x_i)^T S_i^-1 (z_j - H x_i))
```

The Hungarian algorithm selects the minimum-cost assignment under a gating threshold.

## 3. Adaptive EKF

Each track maintains the state:

```text
x_k = [p_x, p_y, v_x, v_y]^T
```

The filter follows a constant-velocity motion model. Covariances are adapted through three tiers:

- Tier 1: measurement noise is scaled by smoothed detection confidence.
- Tier 2: process noise increases during occlusion and missed detections.
- Tier 3: topology-derived feedback updates process and measurement noise.

## 4. Topological Data Analysis

Recent trajectory points are stored in a fixed-size buffer. Persistent homology is computed over this point cloud with a Vietoris-Rips filtration.

The feature vector combines:

- Betti-number and persistence statistics,
- H1 persistence information,
- persistence image descriptors.

The resulting vector is used both for anomaly scoring and topology-aware EKF feedback.

## 5. Anomaly and Crash Visualization

The visual layer is intentionally status-based rather than ID-color-based:

```text
green  -> normal tracked vehicle
yellow -> anomalous vehicle
red    -> crash/collision candidate
gray   -> raw detector bounding box
```

This keeps the output aligned with the research goal: identifying abnormal and critical vehicle behavior rather than emphasizing per-track color variation.
