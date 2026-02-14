import socket
import struct
import cv2
import numpy as np
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import json
import logging
from datetime import datetime
import signal
import sys
import argparse

@dataclass
class StreamStats:
    """Statistics for streaming performance monitoring"""
    fps_values: deque = field(default_factory=lambda: deque(maxlen=100))
    latency_values: deque = field(default_factory=lambda: deque(maxlen=100))
    packet_loss: int = 0
    total_packets: int = 0
    last_frame_id: int = -1
    start_time: float = field(default_factory=time.time)
    
    def update_fps(self, fps: float):
        self.fps_values.append(fps)
    
    def update_latency(self, latency: float):
        self.latency_values.append(latency)
    
    def check_packet_loss(self, frame_id: int):
        self.total_packets += 1
        if self.last_frame_id != -1 and frame_id > self.last_frame_id + 1:
            self.packet_loss += frame_id - self.last_frame_id - 1
        self.last_frame_id = frame_id
    
    def get_avg_fps(self) -> float:
        return np.mean(self.fps_values) if self.fps_values else 0
    
    def get_avg_latency(self) -> float:
        return np.mean(self.latency_values) if self.latency_values else 0
    
    def get_packet_loss_rate(self) -> float:
        if self.total_packets == 0:
            return 0
        return (self.packet_loss / self.total_packets) * 100
    
    def get_uptime(self) -> float:
        return time.time() - self.start_time

class UDPStreamClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 5005, 
                 max_packet_size: int = 65507, 
                 buffer_size: int = 10,
                 enable_recording: bool = False,
                 log_file: Optional[str] = None):
        """
        Enhanced UDP Stream Client
        
        Args:
            ip: IP address to bind to
            port: UDP port to listen on
            max_packet_size: Maximum packet size
            buffer_size: Size of frame buffer for smooth playback
            enable_recording: Enable video recording
            log_file: Path to log file
        """
        self.ip = ip
        self.port = port
        self.max_packet_size = max_packet_size
        self.buffer_size = buffer_size
        self.enable_recording = enable_recording
        self.running = True
        
        # Setup logging
        self.setup_logging(log_file)
        
        # Initialize socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))
        self.sock.settimeout(1.0)  # 1 second timeout
        
        # Statistics
        self.stats = StreamStats()
        
        # Frame buffer for smooth playback
        self.frame_buffer = deque(maxlen=buffer_size)
        self.buffer_lock = threading.Lock()
        
        # Video writer for recording
        self.video_writer = None
        
        # Performance monitoring
        self.prev_time = time.time()
        self.frame_count = 0
        
        # Callback functions
        self.frame_callbacks = []
        
        self.logger.info(f"UDP Stream Client initialized on {self.ip}:{self.port}")
    
    def setup_logging(self, log_file: Optional[str]):
        """Setup logging configuration"""
        self.logger = logging.getLogger('UDPClient')
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
    
    def add_frame_callback(self, callback):
        """Add callback function for frame processing"""
        self.frame_callbacks.append(callback)
    
    def process_frame(self, frame: np.ndarray, frame_id: int, timestamp: float) -> np.ndarray:
        """Process frame with all callbacks"""
        for callback in self.frame_callbacks:
            result = callback(frame, frame_id, timestamp)
            if result is not None:
                frame = result
        return frame
    
    def init_video_writer(self, frame: np.ndarray):
        """Initialize video writer for recording"""
        if self.enable_recording and self.video_writer is None:
            height, width = frame.shape[:2]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))
            self.logger.info(f"Recording to {filename}")
    
    def receive_frame(self) -> Optional[Tuple[np.ndarray, int, float]]:
        """Receive and decode a single frame"""
        try:
            packet, addr = self.sock.recvfrom(self.max_packet_size)
            
            # Parse header
            header_size = struct.calcsize("!IdI")
            frame_id, timestamp, size = struct.unpack("!IdI", packet[:header_size])
            
            # Extract frame data
            data = packet[header_size:]
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            
            if frame is None:
                self.logger.warning(f"Failed to decode frame {frame_id}")
                return None
            
            return frame, frame_id, timestamp
            
        except socket.timeout:
            return None
        except Exception as e:
            self.logger.error(f"Error receiving frame: {e}")
            return None
    
    def display_frame(self, frame: np.ndarray, frame_id: int, latency: float, fps: float):
        """Display frame with enhanced information overlay"""
        display = frame.copy()
        
        # Add information overlay
        height, width = display.shape[:2]
        
        # Semi-transparent overlay for text background
        overlay = display.copy()
        cv2.rectangle(overlay, (10, 10), (350, 140), (0, 0, 0), -1)
        display = cv2.addWeighted(overlay, 0.3, display, 0.7, 0)
        
        # Display information
        info_lines = [
            f"Frame ID: {frame_id}",
            f"FPS: {fps:.1f}",
            f"Latency: {latency*1000:.1f} ms",
            f"Packet Loss: {self.stats.get_packet_loss_rate():.1f}%",
            f"Buffer: {len(self.frame_buffer)}/{self.buffer_size}",
            f"Uptime: {self.stats.get_uptime():.1f}s"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(display, line, (20, 40 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add latency indicator
        latency_ms = latency * 1000
        if latency_ms < 50:
            color = (0, 255, 0)  # Green for good
        elif latency_ms < 150:
            color = (0, 255, 255)  # Yellow for moderate
        else:
            color = (0, 0, 255)  # Red for poor
        
        cv2.circle(display, (width - 50, 50), 10, color, -1)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        cv2.putText(display, timestamp, (width - 250, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display
    
    def buffer_worker(self):
        """Worker thread to receive frames and fill buffer"""
        while self.running:
            result = self.receive_frame()
            if result:
                frame, frame_id, timestamp = result
                with self.buffer_lock:
                    self.frame_buffer.append((frame, frame_id, timestamp))
    
    def run(self):
        """Main client loop"""
        self.logger.info(f"Starting UDP client on {self.ip}:{self.port}")
        
        # Start buffer thread
        buffer_thread = threading.Thread(target=self.buffer_worker, daemon=True)
        buffer_thread.start()
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Performance monitoring window
        cv2.namedWindow("UDP Client Stream", cv2.WINDOW_NORMAL)
        
        try:
            while self.running:
                # Get frame from buffer
                with self.buffer_lock:
                    if not self.frame_buffer:
                        time.sleep(0.001)
                        continue
                    frame, frame_id, timestamp = self.frame_buffer.popleft()
                
                # Calculate latency
                current_time = time.time()
                latency = current_time - timestamp
                
                # Check for packet loss
                self.stats.check_packet_loss(frame_id)
                self.stats.update_latency(latency)
                
                # Calculate FPS
                fps = 1 / (current_time - self.prev_time) if self.prev_time else 0
                self.prev_time = current_time
                self.stats.update_fps(fps)
                self.frame_count += 1
                
                # Initialize video writer if recording
                if self.enable_recording and self.video_writer is None:
                    self.init_video_writer(frame)
                
                # Process frame with callbacks
                processed_frame = self.process_frame(frame, frame_id, timestamp)
                
                # Record frame if enabled
                if self.video_writer:
                    self.video_writer.write(processed_frame)
                
                # Display frame
                display_frame = self.display_frame(processed_frame, frame_id, latency, fps)
                cv2.imshow("UDP Client Stream", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    self.logger.info("ESC pressed, shutting down...")
                    break
                elif key == ord('r'):  # Toggle recording
                    self.enable_recording = not self.enable_recording
                    if not self.enable_recording and self.video_writer:
                        self.video_writer.release()
                        self.video_writer = None
                    self.logger.info(f"Recording {'enabled' if self.enable_recording else 'disabled'}")
                elif key == ord('s'):  # Save statistics
                    self.save_statistics()
                elif key == ord('c'):  # Clear buffer
                    with self.buffer_lock:
                        self.frame_buffer.clear()
                    self.logger.info("Buffer cleared")
                
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
        finally:
            self.cleanup()
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info("Shutdown signal received")
        self.running = False
    
    def save_statistics(self, filename: Optional[str] = None):
        """Save statistics to JSON file"""
        if filename is None:
            filename = f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        stats_data = {
            'avg_fps': self.stats.get_avg_fps(),
            'avg_latency_ms': self.stats.get_avg_latency() * 1000,
            'packet_loss_rate': self.stats.get_packet_loss_rate(),
            'total_packets': self.stats.total_packets,
            'packet_loss': self.stats.packet_loss,
            'frame_count': self.frame_count,
            'uptime': self.stats.get_uptime(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        self.logger.info(f"Statistics saved to {filename}")
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.video_writer:
            self.video_writer.release()
        self.sock.close()
        cv2.destroyAllWindows()
        
        # Save final statistics
        self.save_statistics()
        
        self.logger.info("Cleanup completed")
        self.logger.info(f"Average FPS: {self.stats.get_avg_fps():.2f}")
        self.logger.info(f"Average Latency: {self.stats.get_avg_latency()*1000:.2f} ms")
        self.logger.info(f"Packet Loss Rate: {self.stats.get_packet_loss_rate():.2f}%")

# Example callback functions
def draw_detection_overlay(frame, frame_id, timestamp):
    """Example callback for drawing detections"""
    # This could be your SORT tracking integration
    return frame

def log_frame_info(frame, frame_id, timestamp):
    """Example callback for logging frame information"""
    latency = time.time() - timestamp
    print(f"Frame {frame_id}: Latency {latency*1000:.1f}ms")
    return frame

def save_frame_callback(save_interval: int = 30):
    """Factory function for frame saving callback"""
    counter = 0
    
    def callback(frame, frame_id, timestamp):
        nonlocal counter
        counter += 1
        if counter % save_interval == 0:
            filename = f"frame_{frame_id}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
        return frame
    
    return callback

def create_network_monitor():
    """Create a network monitoring callback"""
    bandwidth = deque(maxlen=100)
    
    def callback(frame, frame_id, timestamp):
        # Estimate bandwidth (rough approximation)
        frame_size = frame.nbytes
        bandwidth.append(frame_size)
        
        if len(bandwidth) > 1:
            avg_size = np.mean(bandwidth)
            cv2.putText(frame, f"BW: {avg_size/1024:.1f} KB/frame", 
                       (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 0), 1)
        return frame
    
    return callback

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='UDP Video Stream Client')
    parser.add_argument('--ip', type=str, default='0.0.0.0',
                       help='IP address to bind to')
    parser.add_argument('--port', type=int, default=5005,
                       help='UDP port to listen on')
    parser.add_argument('--buffer', type=int, default=10,
                       help='Frame buffer size')
    parser.add_argument('--record', action='store_true',
                       help='Enable video recording')
    parser.add_argument('--log', type=str, default=None,
                       help='Log file path')
    parser.add_argument('--stats-interval', type=int, default=60,
                       help='Statistics save interval in seconds')
    
    args = parser.parse_args()
    
    # Create and configure client
    client = UDPStreamClient(
        ip=args.ip,
        port=args.port,
        buffer_size=args.buffer,
        enable_recording=args.record,
        log_file=args.log
    )
    
    # Add example callbacks
    client.add_frame_callback(draw_detection_overlay)
    client.add_frame_callback(create_network_monitor())
    client.add_frame_callback(save_frame_callback(save_interval=100))
    
    # Start client
    try:
        client.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        client.cleanup()

if __name__ == "__main__":
    main()