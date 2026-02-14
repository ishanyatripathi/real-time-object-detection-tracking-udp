import socket
import cv2
import struct
import time
import torch
import numpy as np
from ultralytics import YOLO
from sort_tracker import Sort
from collections import defaultdict, deque
import threading
import queue
import logging
import json
import psutil
import signal
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


# =========================
# PERFORMANCE STATS CLASS
# =========================

@dataclass
class StreamStats:
    fps_values: deque = field(default_factory=lambda: deque(maxlen=100))
    inference_times: deque = field(default_factory=lambda: deque(maxlen=100))
    total_frames: int = 0
    start_time: float = field(default_factory=time.time)

    def update_fps(self, fps):
        self.fps_values.append(fps)

    def update_inference(self, t):
        self.inference_times.append(t)

    def avg_fps(self):
        return np.mean(self.fps_values) if self.fps_values else 0

    def avg_inference_ms(self):
        return np.mean(self.inference_times) * 1000 if self.inference_times else 0


# =========================
# VIDEO STREAM SERVER
# =========================

class VideoStreamServer:

    def __init__(self,
                 ip="127.0.0.1",
                 port=5005,
                 camera_id=0,
                 width=640,
                 height=360,
                 model="yolov8n.pt",
                 conf=0.4,
                 jpeg_quality=70,
                 target_fps=30):

        self.ip = ip
        self.port = port
        self.width = width
        self.height = height
        self.conf = conf
        self.jpeg_quality = jpeg_quality
        self.target_fps = target_fps

        self.frame_id = 0
        self.running = True

        self.stats = StreamStats()

        self.logger = self.setup_logger()

        # GPU setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")

        # Load YOLO
        self.model = YOLO(model)
        self.model.to(self.device)

        # SORT tracker
        self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

        # UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Camera
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        # Queues
        self.frame_queue = queue.Queue(maxsize=3)
        self.result_queue = queue.Queue(maxsize=3)

        # Track history
        self.track_history = defaultdict(lambda: deque(maxlen=30))

    # =========================
    # LOGGER
    # =========================

    def setup_logger(self):

        logger = logging.getLogger("Server")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            ch = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s")
            ch.setFormatter(formatter)
            logger.addHandler(ch)

        return logger

    # =========================
    # DETECTION THREAD
    # =========================

    def detection_worker(self):

        while self.running:

            try:
                frame = self.frame_queue.get(timeout=1)

                start = time.time()

                results = self.model(
                    frame,
                    device=self.device,
                    conf=self.conf,
                    verbose=False
                )[0]

                inference_time = time.time() - start
                self.stats.update_inference(inference_time)

                detections = []

                if results.boxes is not None:

                    boxes = results.boxes.xyxy.cpu().numpy()
                    scores = results.boxes.conf.cpu().numpy()

                    for box, score in zip(boxes, scores):
                        x1, y1, x2, y2 = box
                        detections.append([x1, y1, x2, y2, score])

                detections = np.array(detections) if len(
                    detections) > 0 else np.empty((0, 5))

                tracks = self.tracker.update(detections)

                annotated = self.draw_tracks(frame, tracks)

                self.result_queue.put((annotated, len(tracks)))

            except queue.Empty:
                continue

    # =========================
    # DRAW TRACKS
    # =========================

    def draw_tracks(self, frame, tracks):

        for track in tracks:

            x1, y1, x2, y2, track_id = map(int, track[:5])

            color = (
                (track_id * 50) % 255,
                (track_id * 80) % 255,
                (track_id * 120) % 255
            )

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(frame,
                        f"ID {track_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            self.track_history[track_id].append((cx, cy))

            pts = self.track_history[track_id]

            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], color, 2)

        return frame

    # =========================
    # ENCODE FRAME
    # =========================

    def encode_frame(self, frame):

        encode_param = [
            int(cv2.IMWRITE_JPEG_QUALITY),
            self.jpeg_quality
        ]

        _, buffer = cv2.imencode(".jpg", frame, encode_param)

        return buffer.tobytes()

    # =========================
    # SEND UDP FRAME
    # =========================

    def send_frame(self, data):

        header = struct.pack(
            "!IdI",
            self.frame_id,
            time.time(),
            len(data)
        )

        self.sock.sendto(header + data, (self.ip, self.port))

    # =========================
    # MAIN LOOP
    # =========================

    def run(self):

        self.logger.info("Server started")

        worker = threading.Thread(
            target=self.detection_worker,
            daemon=True
        )

        worker.start()

        prev = time.time()

        while self.running:

            ret, frame = self.cap.read()

            if not ret:
                continue

            frame = cv2.resize(frame, (self.width, self.height))

            if not self.frame_queue.full():
                self.frame_queue.put(frame)

            if not self.result_queue.empty():

                frame, obj_count = self.result_queue.get()

                now = time.time()

                fps = 1 / (now - prev)

                prev = now

                self.stats.update_fps(fps)

                self.overlay_info(frame, fps, obj_count)

                data = self.encode_frame(frame)

                self.send_frame(data)

                self.frame_id += 1

                self.stats.total_frames += 1

            if self.frame_id % 30 == 0:

                self.logger.info(
                    f"FPS: {self.stats.avg_fps():.2f} | "
                    f"Inference: {self.stats.avg_inference_ms():.2f} ms | "
                    f"CPU: {psutil.cpu_percent()}%"
                )

    # =========================
    # OVERLAY INFO
    # =========================

    def overlay_info(self, frame, fps, count):

        cv2.putText(frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)

        cv2.putText(frame,
                    f"Objects: {count}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)

    # =========================
    # CLEANUP
    # =========================

    def stop(self):

        self.running = False
        self.cap.release()
        self.sock.close()


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    server = VideoStreamServer(
        ip="127.0.0.1",
        port=5005,
        camera_id=0,
        width=640,
        height=360,
        model="yolov8n.pt",
        conf=0.4,
        jpeg_quality=70,
        target_fps=30
    )

    try:
        server.run()

    except KeyboardInterrupt:

        server.stop()
        print("Server stopped")
