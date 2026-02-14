# Real-Time Object Detection, Tracking, and UDP Video Streaming System

## Overview

This project implements a real-time object detection and multi-object tracking system with UDP video streaming. The system detects objects using a deep learning model, tracks them across frames with persistent IDs, and streams the annotated video to a remote client over UDP.

This project was developed as part of a Computer Vision Technical Assessment.

Key capabilities include:

* Real-time object detection using YOLOv8
* Multi-object tracking using SORT algorithm
* Persistent object IDs across frames
* Trajectory visualization
* UDP video streaming (server → client)
* Real-time performance metrics (FPS, latency, object count)
* Recording, snapshot capture, and statistics logging

---

# System Architecture

The system consists of four main components:

1. Detection Module
2. Tracking Module
3. UDP Streaming Server
4. UDP Streaming Client

Flow:

Camera → Detection → Tracking → Overlay → UDP Server → Network → UDP Client → Display

---

# Features

## Object Detection

* Uses YOLOv8 (Ultralytics)
* Detects multiple object classes (person, car, bicycle, etc.)
* Displays:

  * Bounding boxes
  * Class labels
  * Confidence scores
* GPU acceleration using CUDA
* Maintains real-time performance (>15 FPS)

## Multi-Object Tracking

* SORT (Simple Online and Realtime Tracking)
* Assigns unique persistent IDs
* Handles:

  * Occlusions
  * Entry and exit
  * Multiple objects simultaneously
* Trajectory visualization (last 30 frames)
* Track lifecycle management

## UDP Video Streaming

Server:

* Captures processed frames
* Compresses frames using JPEG
* Sends frames over UDP
* Includes metadata:

  * frame_id
  * timestamp
  * frame_size

Client:

* Receives UDP packets
* Reconstructs frames
* Displays real-time video
* Shows metrics:

  * FPS
  * latency
  * object count

## Performance Monitoring

Displays:

* FPS
* inference time
* detection count
* track count
* uptime

Supports:

* recording video
* saving snapshots
* exporting statistics

---

# Project Structure

```
drone_assignment/
│
├── yolo_tracking.py        # detection + tracking
├── sort_tracker.py        # SORT algorithm
├── stream_server.py      # UDP streaming server
├── stream_client.py      # UDP streaming client
│
├── README.md
└── requirements.txt
```

---

# Installation

## 1. Clone Repository

```
git clone https://github.com/ishanyatripathi/real-time-object-detection-tracking-udp.git
cd real-time-object-detection-tracking-udp
```

## 2. Create Virtual Environment

```
python -m venv venv
venv\Scripts\activate
```

## 3. Install Dependencies

```
pip install ultralytics
pip install opencv-python
pip install numpy
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install scipy
```

## 4. Download YOLOv8 Model

Automatically downloads when first run:

```
yolov8n.pt
```

---

# Usage

## Run Detection and Tracking

```
python yolo_tracking.py --source 0
```

Options:

```
--source        camera index or video file
--model         yolov8n.pt (default)
--conf          confidence threshold
--iou           IOU threshold
--device        cuda or cpu
--record        enable recording
```

Example:

```
python yolo_tracking.py --source 0 --device cuda --record
```

---

# UDP Streaming

## Start Server

```
python stream_server.py
```

## Start Client

```
python stream_client.py
```

The client will display the streamed video in real-time.

---

# Configuration

## Detection Configuration

Edit in yolo_tracking.py:

```
model_name = "yolov8n.pt"
conf_threshold = 0.5
```

## Tracking Configuration

```
max_age = 20
min_hits = 3
history_length = 30
```

## Streaming Configuration

```
UDP IP
UDP PORT
packet size
compression quality
```

---

# System Design Explanation

This system uses YOLOv8 for object detection because it provides an excellent balance between speed and accuracy. The YOLOv8n (nano) model was chosen specifically for real-time performance on standard GPU hardware while maintaining sufficient detection accuracy.

For tracking, the SORT algorithm was selected due to its lightweight architecture and high efficiency. SORT uses a Kalman Filter for motion prediction and the Hungarian Algorithm for optimal object association between frames. This allows persistent ID assignment, robust handling of temporary occlusions, and reliable multi-object tracking without requiring heavy deep feature extraction.

UDP was chosen as the streaming protocol because it provides low latency transmission, which is critical for real-time applications. Frames are compressed using JPEG to reduce bandwidth usage while maintaining acceptable visual quality. Each packet includes metadata such as frame ID and timestamp, allowing the client to reconstruct frames correctly and measure latency.

Key challenges included maintaining real-time performance while running detection, tracking, and streaming simultaneously. This was addressed by using GPU acceleration, lightweight models, efficient tracking algorithms, and optimized frame processing pipelines.

The final system achieves stable real-time detection, tracking, and streaming while maintaining persistent object identities and smooth video transmission.

---

# Test Scenarios Covered

The system successfully handles:

* Multiple objects (5–10 simultaneously)
* Occlusions
* Entry and exit of objects
* Fast and slow motion
* Real-time network streaming

---

# Output Examples

The system produces:

* Annotated video with bounding boxes and IDs
* Trajectory visualization
* Recorded video files
* Snapshot images
* Performance statistics

---

# Performance

Tested on:

GPU: NVIDIA RTX 2050

Performance:

* 15–30 FPS at 720p
* Stable tracking
* Low latency streaming

---

# Controls

During runtime:

```
ESC → Exit
S   → Save snapshot
R   → Toggle recording
C   → Clear track history
P   → Print statistics
```

---

# Future Improvements

* DeepSORT integration
* ROS integration
* Multi-camera support
* Web dashboard
* Edge deployment

---

# Author

Ishanya Tripathi

Electronics and Telecommunication Engineering

Computer Vision and AI Developer

---

# License

This project is for educational and evaluation purposes.

---

# Demo

https://drive.google.com/drive/folders/1CgqatuxGo0cVidvETRAErNdvE27CiXZM?usp=sharing

---
