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
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import psutil
import signal
import sys
from pathlib import Path
import argparse

@dataclass
class StreamStats:
    """Server streaming statistics"""
    fps_values: deque = field(default_factory=lambda: deque(maxlen=100))
    inference_times: deque = field(default_factory=lambda: deque(maxlen=100))
    encoding_times: deque = field(default_factory=lambda: deque(maxlen=100))
    objects_detected: deque = field(default_factory=lambda: deque(maxlen=100))
    total_frames: int = 0
    start_time: float = field(default_factory=time.time)
    
    def update_fps(self, fps: float):
        self.fps_values.append(fps)
    
    def update_inference_time(self, t: float):
        self.inference_times.append(t)
    
    def update_encoding_time(self, t: float):
        self.encoding_times.append(t)
    
    def update_objects(self, count: int):
        self.objects_detected.append(count)
    
    def get_avg_fps(self) -> float:
        return np.mean(self.fps_values) if self.fps_values else 0
    
    def get_avg_inference_time(self) -> float:
        return np.mean(self.inference_times) if self.inference_times else 0
    
    def get_avg_encoding_time(self) -> float:
        return np.mean(self.encoding_times) if self.encoding_times else 0
    
    def get_avg_objects(self) -> float:
        return np.mean(self.objects_detected) if self.objects_detected else 0
    
    def get_uptime(self) -> float:
        return time.time() - self.start_time

class VideoStreamServer:
    def __init__(self, 
                 udp_ip: str = "127.0.0.1",
                 udp_port: int = 5005,
                 camera_id: int = 0,
                 frame_width: int = 640,
                 frame_height: int = 360,
                 model_name: str = "yolov8n.pt",
                 conf_threshold: float = 0.5,
                 jpeg_quality: int = 70,
                 max_packet_size: int = 65507,
                 target_fps: Optional[int] = None,
                 enable_recording: bool = False,
                 log_file: Optional[str] = None):
        
        # Network config
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.max_packet_size = max_packet_size
        
        # Camera config
        self.camera_id = camera_id
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps if target_fps else 0
        
        # Processing config
        self.conf_threshold = conf_threshold
        self.jpeg_quality = jpeg_quality
        self.enable_recording = enable_recording
        
        # State
        self.running = True
        self.frame_id = 0
        self.stats = StreamStats()
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        
        # Setup logging
        self.setup_logging(log_file)
        
        # Initialize socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Initialize model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"Using device: {self.device}")
        self.model = YOLO(model_name)
        self.model.to(self.device)
        
        # Initialize tracker
        self.tracker = Sort(max_age=20, min_hits=3)
        
        # Initialize camera
        self.cap = None
        self.init_camera()
        
        # Video writer for recording
        self.video_writer = None
        
        # Frame processing queue for parallel processing
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        
        # Performance monitoring
        self.prev_frame_time = time.time()
        self.cpu_usage = deque(maxlen=50)
        self.memory_usage = deque(maxlen=50)
        
        self.logger.info(f"Video Stream Server initialized on {self.udp_ip}:{self.udp_port}")
    
    def setup_logging(self, log_file: Optional[str]):
        """Setup logging configuration"""
        self.logger = logging.getLogger('VideoServer')
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        # File handler if specified
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
    
    def init_camera(self):
        """Initialize camera with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            self.cap = cv2.VideoCapture(self.camera_id)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                
                # Try to set FPS if target specified
                if self.target_fps:
                    self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                
                actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                
                self.logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps:.2f}FPS")
                return
            
            self.logger.warning(f"Failed to open camera (attempt {attempt + 1}/{max_retries})")
            time.sleep(1)
        
        raise RuntimeError("Could not open camera after multiple attempts")
    
    def init_video_writer(self):
        """Initialize video writer for recording"""
        if self.enable_recording and self.video_writer is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                filename, fourcc, 30.0, 
                (self.frame_width, self.frame_height)
            )
            self.logger.info(f"Recording to {filename}")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List, int]:
        """
        Process a single frame: detect, track, annotate
        """
        # Run detection
        inference_start = time.time()
        results = self.model(frame, device=self.device, verbose=False)[0]
        inference_time = time.time() - inference_start
        self.stats.update_inference_time(inference_time)
        
        # Prepare detections
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            
            if conf > self.conf_threshold:
                detections.append([x1, y1, x2, y2])
        
        detections = np.array(detections) if len(detections) > 0 else np.empty((0, 4))
        
        # Update tracker
        tracks = self.tracker.update(detections)
        object_count = len(tracks)
        self.stats.update_objects(object_count)
        
        # Annotate frame
        annotated_frame = self.annotate_frame(frame, tracks)
        
        return annotated_frame, tracks, object_count
    
    def annotate_frame(self, frame: np.ndarray, tracks: List) -> np.ndarray:
        """Annotate frame with tracking information"""
        annotated = frame.copy()
        
        # Draw tracks
        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track[:5])
            
            # Generate color based on track_id
            color = self.get_track_color(track_id)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw ID with background
            label = f"ID {track_id}"
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(annotated, (x1, y1 - label_height - 10), 
                         (x1 + label_width + 10, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Update track history
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            self.track_history[track_id].append((cx, cy))
            
            # Draw trajectory
            points = list(self.track_history[track_id])
            for i in range(1, len(points)):
                cv2.line(annotated, points[i-1], points[i], color, 2)
        
        return annotated
    
    def get_track_color(self, track_id: int) -> tuple:
        """Generate consistent color for track ID"""
        np.random.seed(track_id)
        return tuple(np.random.randint(0, 255, 3).tolist())
    
    def add_info_overlay(self, frame: np.ndarray, fps: float, 
                        object_count: int) -> np.ndarray:
        """Add information overlay to frame"""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        cv2.rectangle(overlay, (10, 10), (300, 130), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # System info
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        # FPS indicator color
        if fps >= 25:
            fps_color = (0, 255, 0)
        elif fps >= 15:
            fps_color = (0, 255, 255)
        else:
            fps_color = (0, 0, 255)
        
        info_lines = [
            f"FPS: {fps:.1f}",
            f"Objects: {object_count}",
            f"CPU: {cpu_percent:.1f}%",
            f"RAM: {memory_percent:.1f}%",
            f"Frame: {self.frame_id}",
            f"Latency: {self.stats.get_avg_inference_time()*1000:.1f}ms"
        ]
        
        for i, line in enumerate(info_lines):
            color = fps_color if i == 0 else (255, 255, 255)
            cv2.putText(frame, line, (20, 40 + i*20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (w - 200, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def encode_frame(self, frame: np.ndarray) -> Optional[bytes]:
        """Encode frame to JPEG"""
        encoding_start = time.time()
        
        try:
            # Adaptive JPEG quality based on network conditions
            quality = self.jpeg_quality
            if self.stats.get_avg_fps() < 15:
                quality = max(40, quality - 10)  # Reduce quality if FPS drops
            
            _, buffer = cv2.imencode(".jpg", frame, 
                                    [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            
            encoding_time = time.time() - encoding_start
            self.stats.update_encoding_time(encoding_time)
            
            return buffer.tobytes()
        except Exception as e:
            self.logger.error(f"Encoding error: {e}")
            return None
    
    def send_frame(self, data: bytes):
        """Send frame over UDP with chunking if needed"""
        try:
            if len(data) < self.max_packet_size:
                # Single packet
                header = struct.pack("!IdI", self.frame_id, time.time(), len(data))
                self.sock.sendto(header + data, (self.udp_ip, self.udp_port))
            else:
                # Chunk large frames
                self.send_chunked_frame(data)
        except Exception as e:
            self.logger.error(f"Send error: {e}")
    
    def send_chunked_frame(self, data: bytes):
        """Send large frame in chunks"""
        chunk_size = self.max_packet_size - struct.calcsize("!IdII")
        num_chunks = (len(data) + chunk_size - 1) // chunk_size
        
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, len(data))
            chunk = data[start:end]
            
            header = struct.pack("!IdIII", self.frame_id, time.time(), 
                                len(data), i, num_chunks)
            self.sock.sendto(header + chunk, (self.udp_ip, self.udp_port))
    
    def worker_thread(self):
        """Worker thread for parallel processing"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                processed_frame, tracks, count = self.process_frame(frame)
                self.result_queue.put((processed_frame, tracks, count))
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker error: {e}")
    
    def run(self):
        """Main server loop"""
        self.logger.info("Starting video stream server...")
        
        # Start worker thread
        worker = threading.Thread(target=self.worker_thread, daemon=True)
        worker.start()
        
        # Setup signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Performance monitoring
        frame_times = deque(maxlen=30)
        
        try:
            while self.running:
                loop_start = time.time()
                
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame, reinitializing camera...")
                    self.init_camera()
                    continue
                
                # Resize frame
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                
                # Add to processing queue
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    # Skip frame if queue is full
                    continue
                
                # Get processed result
                try:
                    processed_frame, tracks, object_count = self.result_queue.get_nowait()
                except queue.Empty:
                    # Use original frame if processing not ready
                    processed_frame = frame
                    object_count = 0
                
                # Calculate FPS
                current_time = time.time()
                frame_time = current_time - self.prev_frame_time
                frame_times.append(frame_time)
                fps = 1.0 / np.mean(frame_times) if frame_times else 0
                self.prev_frame_time = current_time
                self.stats.update_fps(fps)
                
                # Add overlay
                display_frame = self.add_info_overlay(
                    processed_frame, fps, object_count
                )
                
                # Record if enabled
                if self.enable_recording:
                    if self.video_writer is None:
                        self.init_video_writer()
                    if self.video_writer:
                        self.video_writer.write(display_frame)
                
                # Encode and send
                encoding_start = time.time()
                data = self.encode_frame(display_frame)
                if data:
                    self.send_frame(data)
                
                # Update stats
                self.stats.total_frames += 1
                self.frame_id += 1
                
                # FPS limiting
                if self.target_fps:
                    elapsed = time.time() - loop_start
                    sleep_time = max(0, self.frame_interval - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
                # Periodic stats logging
                if self.frame_id % 100 == 0:
                    self.log_stats()
                
        except Exception as e:
            self.logger.error(f"Server error: {e}")
        finally:
            self.cleanup()
    
    def log_stats(self):
        """Log performance statistics"""
        self.logger.info(
            f"Stats - FPS: {self.stats.get_avg_fps():.1f}, "
            f"Objects: {self.stats.get_avg_objects():.1f}, "
            f"Inference: {self.stats.get_avg_inference_time()*1000:.1f}ms, "
            f"Encoding: {self.stats.get_avg_encoding_time()*1000:.1f}ms"
        )
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info("Shutdown signal received")
        self.running = False
    
    def save_stats(self, filename: Optional[str] = None):
        """Save statistics to file"""
        if filename is None:
            filename = f"server_stats_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        stats_data = {
            'avg_fps': self.stats.get_avg_fps(),
            'avg_inference_time_ms': self.stats.get_avg_inference_time() * 1000,
            'avg_encoding_time_ms': self.stats.get_avg_encoding_time() * 1000,
            'avg_objects': self.stats.get_avg_objects(),
            'total_frames': self.stats.total_frames,
            'uptime': self.stats.get_uptime(),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(filename, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        self.logger.info(f"Statistics saved to {filename}")
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        
        if self.cap:
            self.cap.release()
        
        if self.video_writer:
            self.video_writer.release()
        
        self.sock.close()
        
        # Save final stats
        self.save_stats()
        
        cv2.destroyAllWindows()
        self.logger.info("Cleanup completed")

class ServerConfig:
    """Configuration management for server"""
    def __init__(self, config_file: Optional[str] = None):
        self.config = {
            'udp_ip': '127.0.0.1',
            'udp_port': 5005,
            'camera_id': 0,
            'frame_width': 640,
            'frame_height': 360,
            'model_name': 'yolov8n.pt',
            'conf_threshold': 0.5,
            'jpeg_quality': 70,
            'target_fps': 30,
            'enable_recording': False,
            'log_file': None
        }
        
        if config_file and Path(config_file).exists():
            self.load(config_file)
    
    def load(self, config_file: str):
        """Load configuration from file"""
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
            self.config.update(loaded_config)
    
    def save(self, config_file: str):
        """Save configuration to file"""
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def update_from_args(self, args):
        """Update configuration from command line arguments"""
        for key, value in vars(args).items():
            if value is not None and key in self.config:
                self.config[key] = value
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.config.get(key, default)

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='UDP Video Streaming Server')
    
    # Network options
    parser.add_argument('--ip', type=str, help='UDP IP address')
    parser.add_argument('--port', type=int, help='UDP port')
    
    # Camera options
    parser.add_argument('--camera', type=int, help='Camera ID')
    parser.add_argument('--width', type=int, help='Frame width')
    parser.add_argument('--height', type=int, help='Frame height')
    parser.add_argument('--fps', type=int, help='Target FPS')
    
    # Model options
    parser.add_argument('--model', type=str, help='YOLO model name')
    parser.add_argument('--conf', type=float, help='Confidence threshold')
    
    # Encoding options
    parser.add_argument('--quality', type=int, help='JPEG quality (0-100)')
    
    # Other options
    parser.add_argument('--record', action='store_true', help='Enable recording')
    parser.add_argument('--log', type=str, help='Log file path')
    parser.add_argument('--config', type=str, help='Configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = ServerConfig(args.config)
    config.update_from_args(args)
    
    # Create and start server
    server = VideoStreamServer(
        udp_ip=config.get('udp_ip'),
        udp_port=config.get('udp_port'),
        camera_id=config.get('camera_id'),
        frame_width=config.get('frame_width'),
        frame_height=config.get('frame_height'),
        model_name=config.get('model_name'),
        conf_threshold=config.get('conf_threshold'),
        jpeg_quality=config.get('jpeg_quality'),
        target_fps=config.get('target_fps'),
        enable_recording=config.get('enable_recording'),
        log_file=config.get('log_file')
    )
    
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.cleanup()

if __name__ == "__main__":
    main()