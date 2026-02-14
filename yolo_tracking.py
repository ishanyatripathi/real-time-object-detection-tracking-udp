from ultralytics import YOLO
import cv2
import torch
import numpy as np
from collections import defaultdict, deque
from sort_tracker import Sort
import time
import argparse
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import json
from pathlib import Path
import signal
import sys

@dataclass
class TrackingStats:
    """Statistics for tracking performance"""
    fps_values: deque = field(default_factory=lambda: deque(maxlen=100))
    inference_times: deque = field(default_factory=lambda: deque(maxlen=100))
    detection_counts: deque = field(default_factory=lambda: deque(maxlen=100))
    track_counts: deque = field(default_factory=lambda: deque(maxlen=100))
    total_frames: int = 0
    start_time: float = field(default_factory=time.time)
    
    def update_fps(self, fps: float):
        self.fps_values.append(fps)
    
    def update_inference_time(self, t: float):
        self.inference_times.append(t)
    
    def update_detection_count(self, count: int):
        self.detection_counts.append(count)
    
    def update_track_count(self, count: int):
        self.track_counts.append(count)
    
    def get_avg_fps(self) -> float:
        return np.mean(self.fps_values) if self.fps_values else 0
    
    def get_avg_inference_time(self) -> float:
        return np.mean(self.inference_times) if self.inference_times else 0
    
    def get_avg_detections(self) -> float:
        return np.mean(self.detection_counts) if self.detection_counts else 0
    
    def get_avg_tracks(self) -> float:
        return np.mean(self.track_counts) if self.track_counts else 0
    
    def get_uptime(self) -> float:
        return time.time() - self.start_time

class ObjectTracker:
    def __init__(self,
                 model_name: str = "yolov8n.pt",
                 conf_threshold: float = 0.5,
                 iou_threshold: float = 0.5,
                 device: Optional[str] = None,
                 classes: Optional[List[int]] = None,
                 max_age: int = 20,
                 min_hits: int = 3,  # Fixed: min_hits (not min_hints)
                 track_history_length: int = 30,
                 enable_recording: bool = False,
                 output_dir: str = "outputs",
                 log_level: str = "INFO",
                 half_precision: bool = True,  # Added half precision support
                 warmup_frames: int = 10):  # Added warmup frames
        
        # Configuration
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes
        self.track_history_length = track_history_length
        self.enable_recording = enable_recording
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.warmup_frames = warmup_frames
        self.half_precision = half_precision
        
        # Setup logging
        self.setup_logging(log_level)
        
        # Initialize device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.logger.info(f"Using device: {self.device}")
        
        # Check for half precision support
        if self.half_precision and self.device == 'cuda':
            # Check if GPU supports half precision (all RTX cards do)
            self.half_precision = True
            self.logger.info("Half precision enabled for faster inference")
        else:
            self.half_precision = False
        
        # Initialize YOLO model
        self.model = YOLO(model_name)
        self.model.to(self.device)
        
        # Initialize SORT tracker - FIXED: min_hits parameter name
        self.tracker = Sort(max_age=max_age, min_hits=min_hits)
        
        # Tracking history
        self.track_history = defaultdict(lambda: deque(maxlen=track_history_length))
        self.track_colors = {}
        self.track_classes = {}
        self.track_confidences = {}
        self.last_seen_frame = {}  # Track when each ID was last seen
        
        # Performance statistics
        self.stats = TrackingStats()
        
        # Video writer for recording
        self.video_writer = None
        
        # State
        self.running = True
        self.frame_count = 0
        self.prev_time = time.time()
        self.warmed_up = False
        
        # Setup signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        
        self.logger.info("ObjectTracker initialized")
    
    def setup_logging(self, log_level: str):
        """Setup logging configuration"""
        self.logger = logging.getLogger('ObjectTracker')
        self.logger.setLevel(getattr(logging, log_level))
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        # File handler
        fh = logging.FileHandler(self.output_dir / 'tracking.log')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
    
    def warmup(self):
        """Run warmup inference to stabilize performance metrics"""
        self.logger.info(f"Running warmup for {self.warmup_frames} frames...")
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        for i in range(self.warmup_frames):
            # Use half precision if enabled
            if self.half_precision:
                _ = self.model(dummy_frame, device=self.device, verbose=False, half=True)
            else:
                _ = self.model(dummy_frame, device=self.device, verbose=False)
        
        self.logger.info("Warmup complete")
        self.warmed_up = True
    
    def get_track_color(self, track_id: int) -> tuple:
        """Generate consistent color for track ID"""
        if track_id not in self.track_colors:
            # Generate random color with fixed seed for consistency
            np.random.seed(track_id)
            self.track_colors[track_id] = tuple(
                np.random.randint(0, 255, 3).tolist()
            )
        return self.track_colors[track_id]
    
    def associate_detections_with_tracks(self, tracks: np.ndarray, 
                                        detections: List,
                                        detection_classes: List[int],
                                        detection_confidences: List[float]) -> Dict:
        """
        Properly associate detections with tracks using IOU matching
        
        Args:
            tracks: Array of tracks from SORT [x1,y1,x2,y2,track_id]
            detections: List of detection boxes
            detection_classes: List of class IDs for detections
            detection_confidences: List of confidences for detections
        
        Returns:
            Dictionary mapping track_id to (class_id, confidence)
        """
        track_info = {}
        
        if len(tracks) == 0 or len(detections) == 0:
            return track_info
        
        # Calculate IOU matrix between tracks and detections
        iou_matrix = np.zeros((len(tracks), len(detections)))
        
        for t, track in enumerate(tracks):
            track_box = track[:4]
            for d, det in enumerate(detections):
                iou_matrix[t, d] = self.calculate_iou(track_box, det)
        
        # Find best matches (greedy assignment)
        used_detections = set()
        
        for t in range(len(tracks)):
            if len(detections) == 0:
                break
            
            # Find best matching detection for this track
            best_d = np.argmax(iou_matrix[t])
            best_iou = iou_matrix[t, best_d]
            
            # Only associate if IOU is significant (> 0.3)
            if best_iou > 0.3 and best_d not in used_detections:
                track_id = int(tracks[t][4])
                track_info[track_id] = {
                    'class': detection_classes[best_d],
                    'confidence': detection_confidences[best_d]
                }
                used_detections.add(best_d)
                iou_matrix[:, best_d] = -1  # Mark detection as used
        
        return track_info
    
    def calculate_iou(self, box1: np.ndarray, box2: List) -> float:
        """Calculate IOU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        iou = intersection / (area1 + area2 - intersection + 1e-6)
        return iou
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List, Dict]:
        """
        Process a single frame: detect, track, annotate
        """
        # Run inference with optional half precision
        inference_start = time.time()
        
        inference_kwargs = {
            'device': self.device,
            'verbose': False,
            'conf': self.conf_threshold,
            'iou': self.iou_threshold,
            'classes': self.classes
        }
        
        # Add half precision for CUDA if enabled and warmed up
        if self.half_precision and self.warmed_up and self.device == 'cuda':
            inference_kwargs['half'] = True
        
        results = self.model(frame, **inference_kwargs)[0]
        
        inference_time = time.time() - inference_start
        self.stats.update_inference_time(inference_time)
        
        # Prepare detections with class information - vectorized for efficiency
        if len(results.boxes) > 0:
            # Vectorized extraction for better performance
            boxes_xyxy = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            cls_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            detections = boxes_xyxy.tolist()
            detection_classes = cls_ids.tolist()
            detection_confidences = confs.tolist()
        else:
            detections = []
            detection_classes = []
            detection_confidences = []
        
        self.stats.update_detection_count(len(detections))
        
        # Convert to numpy array for tracker
        detections_array = np.array(detections) if detections else np.empty((0, 4))
        
        # Update tracker
        tracks = self.tracker.update(detections_array)
        self.stats.update_track_count(len(tracks))
        
        # Properly associate detections with tracks
        track_info = self.associate_detections_with_tracks(
            tracks, detections, detection_classes, detection_confidences
        )
        
        # Update track metadata and history
        current_frame = self.frame_count
        active_track_ids = set()
        
        for track in tracks:
            track_id = int(track[4])
            active_track_ids.add(track_id)
            self.last_seen_frame[track_id] = current_frame
            
            # Update class info if available from association
            if track_id in track_info:
                self.track_classes[track_id] = track_info[track_id]['class']
                self.track_confidences[track_id] = track_info[track_id]['confidence']
            
            # Update track history
            x1, y1, x2, y2 = map(int, track[:4])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            self.track_history[track_id].append((cx, cy))
        
        # Clean up old track histories (not seen for > max_age frames)
        self.cleanup_old_tracks(current_frame)
        
        # Annotate frame
        annotated_frame = self.annotate_frame(frame, tracks, track_info)
        
        return annotated_frame, tracks, track_info
    
    def cleanup_old_tracks(self, current_frame: int):
        """Remove track histories for old, inactive tracks"""
        max_age_frames = self.tracker.max_age * 2  # Give some buffer
        
        old_tracks = [
            track_id for track_id, last_seen in self.last_seen_frame.items()
            if current_frame - last_seen > max_age_frames
        ]
        
        for track_id in old_tracks:
            if track_id in self.track_history:
                del self.track_history[track_id]
            if track_id in self.track_colors:
                del self.track_colors[track_id]
            if track_id in self.track_classes:
                del self.track_classes[track_id]
            if track_id in self.track_confidences:
                del self.track_confidences[track_id]
            if track_id in self.last_seen_frame:
                del self.last_seen_frame[track_id]
    
    def annotate_frame(self, frame: np.ndarray, tracks: List, 
                      track_info: Dict) -> np.ndarray:
        """Annotate frame with tracking information"""
        annotated = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw tracks
        for track in tracks:
            track_id = int(track[4])
            x1, y1, x2, y2 = map(int, track[:4])
            
            # Get track color
            color = self.get_track_color(track_id)
            
            # Get class info (from stored data if not in current track_info)
            if track_id in track_info:
                class_id = track_info[track_id]['class']
                confidence = track_info[track_id]['confidence']
            else:
                class_id = self.track_classes.get(track_id, -1)
                confidence = self.track_confidences.get(track_id, 0)
            
            # Draw bounding box with varying thickness based on confidence
            thickness = max(1, int(confidence * 3))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label
            if class_id >= 0:
                class_name = self.model.names[class_id]
                label = f"ID:{track_id} {class_name} {confidence:.2f}"
            else:
                label = f"ID:{track_id}"
            
            # Draw label with background
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Label background
            cv2.rectangle(
                annotated,
                (x1, y1 - label_height - 10),
                (x1 + label_width + 10, y1),
                color,
                -1
            )
            
            # Label text
            cv2.putText(
                annotated, label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2
            )
            
            # Draw trajectory
            points = list(self.track_history[track_id])
            for i in range(1, len(points)):
                cv2.line(
                    annotated,
                    points[i-1],
                    points[i],
                    color,
                    2
                )
            
            # Draw direction arrow for moving objects
            if len(points) >= 5:
                # Calculate velocity
                recent_points = points[-5:]
                dx = recent_points[-1][0] - recent_points[0][0]
                dy = recent_points[-1][1] - recent_points[0][1]
                speed = np.sqrt(dx**2 + dy**2)
                
                if speed > 10:  # Only show arrow for moving objects
                    arrow_start = points[-2]
                    arrow_end = points[-1]
                    cv2.arrowedLine(
                        annotated,
                        arrow_start,
                        arrow_end,
                        color,
                        2,
                        tipLength=0.3
                    )
        
        return annotated
    
    def add_info_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add information overlay to frame"""
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        # Semi-transparent background for stats
        cv2.rectangle(overlay, (10, 10), (350, 160), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Calculate current FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self.prev_time)
        self.prev_time = current_time
        self.stats.update_fps(fps)
        
        # FPS color coding
        if fps >= 25:
            fps_color = (0, 255, 0)
        elif fps >= 15:
            fps_color = (0, 255, 255)
        else:
            fps_color = (0, 0, 255)
        
        # Get active tracks (seen in last frame)
        active_tracks = len([t for t in self.tracker.tracks if t.no_losses == 0])
        
        # Statistics lines
        stats_lines = [
            f"FPS: {fps:.1f}",
            f"Frame: {self.frame_count}",
            f"Detections: {self.stats.detection_counts[-1] if self.stats.detection_counts else 0}",
            f"Active Tracks: {active_tracks}",
            f"Total Tracks: {len(self.track_history)}",
            f"Inference: {self.stats.get_avg_inference_time()*1000:.1f}ms",
            f"Uptime: {self.stats.get_uptime():.1f}s"
        ]
        
        for i, line in enumerate(stats_lines):
            color = fps_color if i == 0 else (255, 255, 255)
            cv2.putText(
                frame, line,
                (20, 40 + i*20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1
            )
        
        # Add half precision indicator
        if self.half_precision and self.device == 'cuda':
            cv2.putText(
                frame, "FP16",
                (width - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 255), 1
            )
        
        # Add timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            frame, timestamp,
            (width - 200, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 1
        )
        
        return frame
    
    def init_video_writer(self, frame: np.ndarray):
        """Initialize video writer for recording"""
        if self.enable_recording and self.video_writer is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = self.output_dir / f"tracking_{timestamp}.mp4"
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                str(filename), fourcc, 30.0, (width, height)
            )
            self.logger.info(f"Recording to {filename}")
    
    def save_snapshot(self, frame: np.ndarray, tracks: List):
        """Save a snapshot with tracking information"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"snapshot_{timestamp}.jpg"
        cv2.imwrite(str(filename), frame)
        
        # Save tracking info
        info_filename = self.output_dir / f"snapshot_{timestamp}.json"
        track_data = []
        for track in tracks:
            track_id = int(track[4])
            track_data.append({
                'track_id': track_id,
                'bbox': track[:4].tolist(),
                'class': self.track_classes.get(track_id, -1),
                'class_name': self.model.names.get(
                    self.track_classes.get(track_id, -1), 'unknown'
                ),
                'confidence': self.track_confidences.get(track_id, 0)
            })
        
        with open(info_filename, 'w') as f:
            json.dump({
                'frame': self.frame_count,
                'timestamp': timestamp,
                'tracks': track_data
            }, f, indent=2)
        
        self.logger.info(f"Snapshot saved: {filename}")
    
    def run(self, source: str = "0"):
        """
        Main tracking loop
        
        Args:
            source: Video source (camera index, video file path, or stream URL)
        """
        self.logger.info(f"Starting tracking from source: {source}")
        
        # Run warmup before main loop
        self.warmup()
        
        # Open video source
        if source.isdigit():
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            self.logger.error("Failed to open video source")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.logger.info(f"Video properties: {width}x{height}, {total_frames} frames")
        
        # Create display window
        cv2.namedWindow("YOLO + SORT Tracking", cv2.WINDOW_NORMAL)
        
        try:
            while self.running:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    self.logger.info("End of video source")
                    break
                
                self.frame_count += 1
                self.stats.total_frames += 1  # FIXED: Increment total_frames
                
                # Process frame
                annotated_frame, tracks, track_info = self.process_frame(frame)
                
                # Add info overlay
                display_frame = self.add_info_overlay(annotated_frame)
                
                # Record if enabled
                if self.enable_recording:
                    self.init_video_writer(frame)
                    if self.video_writer:
                        self.video_writer.write(display_frame)
                
                # Display frame
                cv2.imshow("YOLO + SORT Tracking", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    self.logger.info("ESC pressed, stopping...")
                    break
                elif key == ord('s'):  # Save snapshot
                    self.save_snapshot(display_frame, tracks)
                elif key == ord('r'):  # Toggle recording
                    self.enable_recording = not self.enable_recording
                    if not self.enable_recording and self.video_writer:
                        self.video_writer.release()
                        self.video_writer = None
                    self.logger.info(f"Recording {'enabled' if self.enable_recording else 'disabled'}")
                elif key == ord('c'):  # Clear track history
                    self.track_history.clear()
                    self.logger.info("Track history cleared")
                elif key == ord('p'):  # Print stats
                    self.print_stats()
                
                # Periodic logging
                if self.frame_count % 100 == 0:
                    self.logger.info(
                        f"Frame {self.frame_count}/{total_frames if total_frames > 0 else '?'} - "
                        f"FPS: {self.stats.get_avg_fps():.1f}, "
                        f"Tracks: {len(tracks)}"
                    )
        
        except Exception as e:
            self.logger.error(f"Error in tracking loop: {e}")
        finally:
            self.cleanup(cap)
    
    def print_stats(self):
        """Print current statistics"""
        self.logger.info("=" * 50)
        self.logger.info("TRACKING STATISTICS")
        self.logger.info("=" * 50)
        self.logger.info(f"Average FPS: {self.stats.get_avg_fps():.2f}")
        self.logger.info(f"Average Inference Time: {self.stats.get_avg_inference_time()*1000:.2f}ms")
        self.logger.info(f"Average Detections: {self.stats.get_avg_detections():.1f}")
        self.logger.info(f"Average Tracks: {self.stats.get_avg_tracks():.1f}")
        self.logger.info(f"Total Frames: {self.stats.total_frames}")  # Now shows correct count
        self.logger.info(f"Total Unique Tracks: {len(self.track_history)}")
        self.logger.info(f"Uptime: {self.stats.get_uptime():.1f}s")
        self.logger.info("=" * 50)
    
    def save_stats(self, filename: Optional[str] = None):
        """Save statistics to file"""
        if filename is None:
            filename = self.output_dir / f"stats_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        stats_data = {
            'avg_fps': self.stats.get_avg_fps(),
            'avg_inference_time_ms': self.stats.get_avg_inference_time() * 1000,
            'avg_detections': self.stats.get_avg_detections(),
            'avg_tracks': self.stats.get_avg_tracks(),
            'total_frames': self.stats.total_frames,  # Now correct
            'total_unique_tracks': len(self.track_history),
            'uptime': self.stats.get_uptime(),
            'half_precision': self.half_precision,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(filename, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        self.logger.info(f"Statistics saved to {filename}")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info("Shutdown signal received")
        self.running = False
    
    def cleanup(self, cap=None):
        """Clean up resources"""
        self.running = False
        
        if cap:
            cap.release()
        
        if self.video_writer:
            self.video_writer.release()
        
        cv2.destroyAllWindows()
        
        # Save final statistics
        self.save_stats()
        self.print_stats()
        
        self.logger.info("Cleanup completed")

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='YOLO + SORT Object Tracking')
    
    # Video source
    parser.add_argument('--source', type=str, default='0',
                       help='Video source (camera index, video file, or stream URL)')
    
    # Model options
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLO model name or path')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5,
                       help='IOU threshold for NMS')
    parser.add_argument('--classes', type=int, nargs='+',
                       help='Filter by classes (e.g., --classes 0 2 for person and car)')
    parser.add_argument('--device', type=str,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--no-half', action='store_true',
                       help='Disable half precision inference')
    
    # Tracker options
    parser.add_argument('--max-age', type=int, default=20,
                       help='Maximum age for lost tracks')
    parser.add_argument('--min-hits', type=int, default=3,  # Fixed parameter name
                       help='Minimum hits to confirm track')
    parser.add_argument('--history-length', type=int, default=30,
                       help='Track history length')
    
    # Output options
    parser.add_argument('--record', action='store_true',
                       help='Enable video recording')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory for recordings and stats')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Create tracker instance
    tracker = ObjectTracker(
        model_name=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        classes=args.classes,
        max_age=args.max_age,
        min_hits=args.min_hits,  # Fixed parameter name
        track_history_length=args.history_length,
        enable_recording=args.record,
        output_dir=args.output_dir,
        log_level=args.log_level,
        half_precision=not args.no_half  # Enable half precision by default
    )
    
    # Run tracking
    tracker.run(source=args.source)

if __name__ == "__main__":
    main()