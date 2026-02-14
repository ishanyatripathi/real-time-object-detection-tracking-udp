import socket
import struct
import cv2
import numpy as np
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple
import json
import logging
from datetime import datetime
import signal
import argparse


# ================================
# Statistics Class
# ================================

@dataclass
class StreamStats:
    fps_values: deque = field(default_factory=lambda: deque(maxlen=100))
    latency_values: deque = field(default_factory=lambda: deque(maxlen=100))

    packet_loss: int = 0
    total_packets: int = 0
    last_frame_id: int = -1

    start_time: float = field(default_factory=time.time)

    def update_fps(self, fps):
        if fps > 0:
            self.fps_values.append(fps)

    def update_latency(self, latency):
        if latency >= 0:
            self.latency_values.append(latency)

    def check_packet_loss(self, frame_id):

        if self.last_frame_id != -1 and frame_id > self.last_frame_id + 1:
            self.packet_loss += frame_id - self.last_frame_id - 1

        self.last_frame_id = frame_id
        self.total_packets += 1

    def get_avg_fps(self):
        return np.mean(self.fps_values) if self.fps_values else 0

    def get_avg_latency(self):
        return np.mean(self.latency_values) if self.latency_values else 0

    def get_packet_loss_rate(self):

        if self.total_packets == 0:
            return 0

        return (self.packet_loss / self.total_packets) * 100

    def get_uptime(self):
        return time.time() - self.start_time


# ================================
# UDP Client
# ================================

class UDPStreamClient:

    def __init__(
        self,
        ip="0.0.0.0",
        port=5005,
        buffer_size=30,
        max_packet_size=65507,
        enable_recording=False,
        log_file=None
    ):

        self.ip = ip
        self.port = port
        self.buffer_size = buffer_size
        self.max_packet_size = max_packet_size
        self.enable_recording = enable_recording

        self.running = True

        self.setup_logging(log_file)

        # Socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((ip, port))
        self.sock.settimeout(1.0)

        self.logger.info(f"UDP Client initialized on {ip}:{port}")

        # Frame buffer
        self.frame_buffer = deque(maxlen=buffer_size)
        self.lock = threading.Lock()

        # Stats
        self.stats = StreamStats()

        # Timing
        self.prev_time = None

        # Video writer
        self.video_writer = None

        self.frame_count = 0


    # ================================
    # Logging
    # ================================

    def setup_logging(self, log_file):

        self.logger = logging.getLogger("UDPClient")

        if not self.logger.handlers:

            self.logger.setLevel(logging.INFO)

            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

            if log_file:
                fh = logging.FileHandler(log_file)
                fh.setFormatter(formatter)
                self.logger.addHandler(fh)


    # ================================
    # Receive Frame
    # ================================

    def receive_frame(self):

        try:

            packet, _ = self.sock.recvfrom(self.max_packet_size)

            header_size = struct.calcsize("!IdI")

            if len(packet) < header_size:
                return None

            frame_id, timestamp, size = struct.unpack(
                "!IdI", packet[:header_size]
            )

            data = packet[header_size:]

            if len(data) != size:
                return None

            frame = cv2.imdecode(
                np.frombuffer(data, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )

            if frame is None:
                return None

            return frame, frame_id, timestamp

        except socket.timeout:
            return None

        except OSError:
            return None

        except Exception as e:
            self.logger.error(f"Receive error: {e}")
            return None


    # ================================
    # Buffer Worker Thread
    # ================================

    def buffer_worker(self):

        while self.running:

            result = self.receive_frame()

            if result:

                with self.lock:
                    self.frame_buffer.append(result)


    # ================================
    # FPS Calculation (FIXED)
    # ================================

    def calculate_fps(self):

        now = time.time()

        if self.prev_time is None:
            self.prev_time = now
            return 0

        delta = now - self.prev_time

        self.prev_time = now

        if delta <= 0:
            return 0

        return 1.0 / delta


    # ================================
    # Display Frame
    # ================================

    def display_frame(self, frame, frame_id, latency, fps):

        display = frame.copy()

        overlay = display.copy()

        cv2.rectangle(overlay, (10, 10), (360, 160), (0, 0, 0), -1)

        display = cv2.addWeighted(overlay, 0.4, display, 0.6, 0)

        lines = [

            f"Frame: {frame_id}",
            f"FPS: {fps:.1f}",
            f"Latency: {latency*1000:.1f} ms",
            f"Packet Loss: {self.stats.get_packet_loss_rate():.1f}%",
            f"Buffer: {len(self.frame_buffer)}",
            f"Uptime: {self.stats.get_uptime():.1f}s"
        ]

        y = 35

        for line in lines:

            cv2.putText(
                display,
                line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            y += 22

        return display


    # ================================
    # Video Writer
    # ================================

    def init_writer(self, frame):

        if self.video_writer is None and self.enable_recording:

            h, w = frame.shape[:2]

            filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            self.video_writer = cv2.VideoWriter(
                filename,
                fourcc,
                30,
                (w, h)
            )

            self.logger.info(f"Recording started: {filename}")


    # ================================
    # Main Loop
    # ================================

    def run(self):

        buffer_thread = threading.Thread(
            target=self.buffer_worker,
            daemon=True
        )

        buffer_thread.start()

        cv2.namedWindow("UDP Stream", cv2.WINDOW_NORMAL)

        signal.signal(signal.SIGINT, self.stop)

        self.logger.info("Client running...")

        try:

            while self.running:

                with self.lock:

                    if not self.frame_buffer:
                        continue

                    frame, frame_id, timestamp = self.frame_buffer.popleft()

                now = time.time()

                latency = now - timestamp

                fps = self.calculate_fps()

                self.stats.update_fps(fps)

                self.stats.update_latency(latency)

                self.stats.check_packet_loss(frame_id)

                self.frame_count += 1

                self.init_writer(frame)

                if self.video_writer:
                    self.video_writer.write(frame)

                display = self.display_frame(frame, frame_id, latency, fps)

                cv2.imshow("UDP Stream", display)

                key = cv2.waitKey(1)

                if key == 27:
                    break

        finally:

            self.cleanup()


    # ================================
    # Shutdown
    # ================================

    def stop(self, *args):

        self.logger.info("Stopping client...")
        self.running = False


    def cleanup(self):

        self.running = False

        try:
            self.sock.close()
        except:
            pass

        if self.video_writer:
            self.video_writer.release()

        cv2.destroyAllWindows()

        self.save_stats()

        self.logger.info("Cleanup done")


    # ================================
    # Save Stats
    # ================================

    def save_stats(self):

        stats = {

            "avg_fps": self.stats.get_avg_fps(),
            "avg_latency_ms": self.stats.get_avg_latency()*1000,
            "packet_loss_rate": self.stats.get_packet_loss_rate(),
            "total_frames": self.frame_count,
            "uptime": self.stats.get_uptime()
        }

        filename = f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filename, "w") as f:
            json.dump(stats, f, indent=4)

        self.logger.info(f"Stats saved: {filename}")


# ================================
# Main
# ================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--buffer", type=int, default=30)
    parser.add_argument("--record", action="store_true")

    args = parser.parse_args()

    client = UDPStreamClient(
        ip=args.ip,
        port=args.port,
        buffer_size=args.buffer,
        enable_recording=args.record
    )

    client.run()


if __name__ == "__main__":
    main()
