import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from collections import deque
import cv2

def iou(bb_test, bb_gt):
    """
    Compute Intersection over Union between two bounding boxes
    Format: [x1, y1, x2, y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    
    area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area_gt = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    
    return wh / (area_test + area_gt - wh + 1e-16)  # Added small epsilon to avoid division by zero

def convert_bbox_to_z(bbox):
    """
    Convert bounding box to Kalman filter measurement space
    [x1, y1, x2, y2] -> [x, y, s, r]
    where s = area, r = aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h  # scale is area
    r = w / h  # aspect ratio
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    """
    Convert Kalman filter state to bounding box
    [x, y, s, r] -> [x1, y1, x2, y2]
    """
    w = np.sqrt(x[2] * x[3])  # width = sqrt(area * aspect_ratio)
    h = x[2] / w  # height = area / width
    x1 = x[0] - w/2.
    y1 = x[1] - h/2.
    x2 = x[0] + w/2.
    y2 = x[1] + h/2.
    
    if score is None:
        return np.array([x1, y1, x2, y2]).reshape((1, 4))
    else:
        return np.array([x1, y1, x2, y2, score]).reshape((1, 5))

class Track:
    count = 0
    
    def __init__(self, bbox, track_id=None, detection_conf=1.0):
        # Define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        
        # Measurement function
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])
        
        # Measurement noise covariance
        self.kf.R[2:,2:] *= 10.
        
        # Process noise covariance
        self.kf.P[4:,4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        
        # Process noise
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        
        # Initialize state
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        
        # Track metadata
        self.track_id = track_id if track_id is not None else Track.get_next_id()
        self.hits = 1
        self.no_losses = 0
        self.age = 0
        self.detection_conf = detection_conf
        self.bbox_history = deque(maxlen=30)  # Store last 30 positions
        self.bbox_history.append(bbox)
        self.color = self._generate_color()
        self.last_velocity = np.zeros(2)
        self.avg_speed = 0.0
        
    @classmethod
    def get_next_id(cls):
        cls.count += 1
        return cls.count
    
    def _generate_color(self):
        """Generate a unique color for visualization"""
        np.random.seed(self.track_id)
        return tuple(np.random.randint(0, 255, 3).tolist())
    
    def predict(self):
        """Advance the state and return the predicted location"""
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        
        self.kf.predict()
        self.age += 1
        
        if self.no_losses > 0:
            self.hits += 1
        
        # Store velocity for speed calculation
        self.last_velocity = self.kf.x[4:6].flatten()
        self.avg_speed = np.linalg.norm(self.last_velocity)
        
        return convert_x_to_bbox(self.kf.x)
    
    def update(self, bbox, detection_conf=1.0):
        """Update the Kalman filter with a new detection"""
        self.no_losses = 0
        self.hits += 1
        self.detection_conf = detection_conf
        self.bbox_history.append(bbox)
        self.kf.update(convert_bbox_to_z(bbox))
    
    def get_state(self):
        """Get current bounding box estimate"""
        return convert_x_to_bbox(self.kf.x).flatten()
    
    def get_velocity(self):
        """Get current velocity (dx, dy)"""
        return self.kf.x[4:6].flatten()
    
    def get_predicted_path(self, steps=5):
        """Predict future positions"""
        future_positions = []
        temp_x = self.kf.x.copy()
        for _ in range(steps):
            temp_x = self.kf.F @ temp_x
            bbox = convert_x_to_bbox(temp_x).flatten()
            future_positions.append(bbox)
        return future_positions

class Sort:
    def __init__(self, max_age=20, min_hits=3, iou_threshold=0.3):
        """
        Initialize SORT tracker
        
        Args:
            max_age: Maximum number of frames to keep track alive without detections
            min_hits: Minimum number of hits to confirm a track
            iou_threshold: IOU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.frame_count = 0
        self.track_history = {}  # Store complete track history for analysis
        
    def _match_detections_to_tracks(self, detections):
        """
        Match detections to existing tracks using Hungarian algorithm
        """
        if len(self.tracks) == 0 or len(detections) == 0:
            return [], list(range(len(detections))), list(range(len(self.tracks)))
        
        # Compute IOU matrix
        iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
        
        for t, track in enumerate(self.tracks):
            track_bbox = track.get_state()
            for d, det in enumerate(detections):
                iou_matrix[t, d] = iou(track_bbox, det)
        
        # Use Hungarian algorithm for optimal assignment
        matched_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = np.array(list(zip(*matched_indices)))
        
        # Filter matches with low IOU
        unmatched_detections = []
        unmatched_tracks = []
        matches = []
        
        for t, d in matched_indices:
            if iou_matrix[t, d] >= self.iou_threshold:
                matches.append((t, d))
            else:
                unmatched_detections.append(d)
                unmatched_tracks.append(t)
        
        # Find all unmatched tracks
        for t in range(len(self.tracks)):
            if t not in matched_indices[:, 0]:
                unmatched_tracks.append(t)
        
        # Find all unmatched detections
        for d in range(len(detections)):
            if d not in matched_indices[:, 1]:
                unmatched_detections.append(d)
        
        return matches, unmatched_detections, unmatched_tracks
    
    def update(self, detections, detection_confs=None):
        """
        Update tracker with new detections
        
        Args:
            detections: List of bounding boxes [x1, y1, x2, y2]
            detection_confs: Optional list of detection confidences
        
        Returns:
            List of tracked objects [x1, y1, x2, y2, track_id, confidence]
        """
        self.frame_count += 1
        
        if detection_confs is None:
            detection_confs = [1.0] * len(detections)
        
        # Get predicted locations from existing tracks
        for track in self.tracks:
            track.predict()
        
        # Match detections to tracks
        matches, unmatched_detections, unmatched_tracks = self._match_detections_to_tracks(detections)
        
        # Update matched tracks
        for t_idx, d_idx in matches:
            self.tracks[t_idx].update(detections[d_idx], detection_confs[d_idx])
        
        # Create new tracks for unmatched detections
        for d_idx in unmatched_detections:
            new_track = Track(detections[d_idx], detection_conf=detection_confs[d_idx])
            self.tracks.append(new_track)
        
        # Mark unmatched tracks as lost
        for t_idx in unmatched_tracks:
            self.tracks[t_idx].no_losses += 1
        
        # Remove old tracks
        self.tracks = [t for t in self.tracks if t.no_losses <= self.max_age]
        
        # Store track history
        for track in self.tracks:
            if track.track_id not in self.track_history:
                self.track_history[track.track_id] = []
            state = track.get_state()
            self.track_history[track.track_id].append({
                'frame': self.frame_count,
                'bbox': state,
                'velocity': track.get_velocity().tolist()
            })
        
        # Prepare output
        output = []
        for track in self.tracks:
            # Only output tracks that have been seen enough times
            if track.hits >= self.min_hits or self.frame_count <= self.min_hits:
                bbox = track.get_state()
                output.append([
                    *bbox.tolist(),
                    track.track_id,
                    track.detection_conf,
                    track.avg_speed
                ])
        
        return np.array(output)
    
    def get_track(self, track_id):
        """Get a specific track by ID"""
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None
    
    def get_active_tracks(self):
        """Get all currently active tracks"""
        return [t for t in self.tracks if t.no_losses == 0]
    
    def get_statistics(self):
        """Get tracking statistics"""
        return {
            'total_tracks': len(self.tracks),
            'active_tracks': len(self.get_active_tracks()),
            'lost_tracks': len([t for t in self.tracks if t.no_losses > 0]),
            'avg_track_age': np.mean([t.age for t in self.tracks]) if self.tracks else 0,
            'total_tracked_objects': sum(len(history) for history in self.track_history.values())
        }
    
    def visualize(self, frame, draw_path=True, path_length=10):
        """
        Draw tracks on frame for visualization
        """
        vis_frame = frame.copy()
        
        for track in self.tracks:
            # Draw current bounding box
            bbox = track.get_state().astype(int)
            cv2.rectangle(vis_frame, 
                         (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]), 
                         track.color, 2)
            
            # Draw track ID and confidence
            label = f"ID: {track.track_id} ({track.detection_conf:.2f})"
            cv2.putText(vis_frame, label, 
                       (bbox[0], bbox[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, track.color, 2)
            
            # Draw velocity vector
            if np.linalg.norm(track.last_velocity) > 0:
                center = ((bbox[0] + bbox[2])//2, (bbox[1] + bbox[3])//2)
                velocity = track.last_velocity * 2
                end_point = (int(center[0] + velocity[0]), 
                           int(center[1] + velocity[1]))
                cv2.arrowedLine(vis_frame, center, end_point, track.color, 2)
            
            # Draw path history
            if draw_path and len(track.bbox_history) > 1:
                points = []
                for hist_bbox in track.bbox_history:
                    center = ((hist_bbox[0] + hist_bbox[2])//2, 
                            (hist_bbox[1] + hist_bbox[3])//2)
                    points.append(center)
                
                for i in range(1, len(points)):
                    alpha = i / len(points)
                    color = tuple(int(c * alpha) for c in track.color)
                    cv2.line(vis_frame, points[i-1], points[i], color, 2)
        
        return vis_frame


# Example usage with additional utility functions
def create_detection(bbox, confidence=1.0):
    """Helper function to create detection objects"""
    return {
        'bbox': bbox,
        'confidence': confidence
    }

def load_detections_from_file(file_path):
    """Load detections from file (format: frame, x1, y1, x2, y2, conf)"""
    detections_by_frame = {}
    with open(file_path, 'r') as f:
        for line in f:
            data = line.strip().split(',')
            frame = int(data[0])
            bbox = [float(x) for x in data[1:5]]
            conf = float(data[5])
            
            if frame not in detections_by_frame:
                detections_by_frame[frame] = []
            detections_by_frame[frame].append((bbox, conf))
    
    return detections_by_frame

def process_video(video_path, detections_by_frame):
    """Process video with tracking visualization"""
    cap = cv2.VideoCapture(video_path)
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Get detections for current frame
        if frame_count in detections_by_frame:
            detections = [d[0] for d in detections_by_frame[frame_count]]
            confidences = [d[1] for d in detections_by_frame[frame_count]]
        else:
            detections = []
            confidences = []
        
        # Update tracker
        tracked_objects = tracker.update(detections, confidences)
        
        # Visualize
        vis_frame = tracker.visualize(frame, draw_path=True)
        
        # Display statistics
        stats = tracker.get_statistics()
        cv2.putText(vis_frame, f"Active Tracks: {stats['active_tracks']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('SORT Tracking', vis_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Quick test
if __name__ == "__main__":
    # Create a simple test
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    
    # Simulate 10 frames of detections
    for frame in range(10):
        # Example detections (moving object)
        detections = [
            [100 + frame*2, 100 + frame, 150 + frame*2, 150 + frame]  # Object moving diagonally
        ]
        confidences = [0.95]
        
        tracked = tracker.update(detections, confidences)
        print(f"Frame {frame}: Tracked {len(tracked)} objects")
        
        stats = tracker.get_statistics()
        print(f"  Statistics: {stats}")