import cv2
import mediapipe as mp
import numpy as np
import json
from datetime import datetime
import os

class PoseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.3,  # Lowered from 0.5
            min_tracking_confidence=0.3,   # Lowered from 0.5
            model_complexity=1
        )
        
        # Create directories if they don't exist
        self.data_dir = "shot_data"
        self.reference_dir = os.path.join(self.data_dir, "reference_shots")
        self.user_dir = os.path.join(self.data_dir, "user_shots")
        
        for directory in [self.data_dir, self.reference_dir, self.user_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Initialize stage detection parameters
        self.stages = {
            'stance': {'detected': False, 'frame': None},
            'grip': {'detected': False, 'frame': None},
            'bend': {'detected': False, 'frame': None},
            'upward': {'detected': False, 'frame': None},
            'release': {'detected': False, 'frame': None},
            'follow_through': {'detected': False, 'frame': None},
            'landing': {'detected': False, 'frame': None}
        }
        
        # Motion tracking
        self.prev_frame = None
        self.motion_history = []
        self.height_history = []
        self.knee_angle_history = []

    def extract_pose_data(self, frame):
        """Extract pose landmarks from a single frame"""
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None
            
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })
        return landmarks

    def detect_motion(self, frame1, frame2, threshold=15):  # Lowered from 30
        """Detect if there is significant motion between frames"""
        if frame1 is None or frame2 is None:
            return True
            
        # Convert frames to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        # If the mean difference is above threshold, there is motion
        return np.mean(diff) > threshold

    def detect_stance(self, frame_data):
        """Detect the initial stance position"""
        if not frame_data:
            return False
            
        # Get key points
        right_hip = frame_data[24]
        right_knee = frame_data[26]
        right_ankle = frame_data[28]
        left_hip = frame_data[23]
        left_knee = frame_data[25]
        left_ankle = frame_data[27]
        
        # Calculate knee angles
        right_knee_angle = self._calculate_angle(right_hip, right_knee, right_ankle)
        left_knee_angle = self._calculate_angle(left_hip, left_knee, left_ankle)
        
        # Check if knees are slightly bent (between 150-170 degrees)
        return 150 <= right_knee_angle <= 170 and 150 <= left_knee_angle <= 170

    def detect_grip(self, frame_data):
        """Detect the ball grip position"""
        if not frame_data:
            return False
            
        # Get key points
        right_shoulder = frame_data[12]
        right_elbow = frame_data[14]
        right_wrist = frame_data[16]
        
        # Calculate elbow angle
        elbow_angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # Check if elbow is aligned (around 90 degrees)
        return 80 <= elbow_angle <= 100

    def detect_bend(self, frame_data):
        """Detect the bending position"""
        if not frame_data:
            return False
            
        # Get key points
        right_hip = frame_data[24]
        right_knee = frame_data[26]
        right_ankle = frame_data[28]
        
        # Calculate knee angle
        knee_angle = self._calculate_angle(right_hip, right_knee, right_ankle)
        
        # Check if knees are significantly bent (between 90-120 degrees)
        return 90 <= knee_angle <= 120

    def detect_upward_motion(self, frame_data, prev_frame_data):
        """Detect the upward motion"""
        if not frame_data or not prev_frame_data:
            return False
            
        # Get current and previous wrist positions
        current_wrist = frame_data[16]
        prev_wrist = prev_frame_data[16]
        
        # Calculate vertical velocity
        vertical_velocity = current_wrist['y'] - prev_wrist['y']
        
        # Check for significant upward motion
        return vertical_velocity < -0.01  # Negative because y increases downward

    def detect_release(self, frame_data, prev_frame_data):
        """Detect the release point"""
        if not frame_data or not prev_frame_data:
            return False
            
        # Get current and previous wrist positions
        current_wrist = frame_data[16]
        prev_wrist = prev_frame_data[16]
        
        # Check if wrist is at its highest point
        return current_wrist['y'] > prev_wrist['y']

    def detect_follow_through(self, frame_data):
        """Detect the follow-through position"""
        if not frame_data:
            return False
            
        # Get key points
        right_shoulder = frame_data[12]
        right_elbow = frame_data[14]
        right_wrist = frame_data[16]
        
        # Calculate arm extension
        arm_extension = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # Check if arm is extended (close to 180 degrees)
        return arm_extension >= 160

    def detect_landing(self, frame_data, prev_frame_data):
        """Detect the landing position"""
        if not frame_data or not prev_frame_data:
            return False
            
        # Get current and previous ankle positions
        current_ankle = frame_data[28]
        prev_ankle = prev_frame_data[28]
        
        # Check if vertical movement has stopped
        return abs(current_ankle['y'] - prev_ankle['y']) < 0.001

    def _calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        a = np.array([point1['x'], point1['y']])
        b = np.array([point2['x'], point2['y']])
        c = np.array([point3['x'], point3['y']])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)

    def analyze_shot(self, video_path, is_reference=False, player_name=None):
        """Analyze a shooting video and save the pose data with stages"""
        cap = cv2.VideoCapture(video_path)
        frames_data = []
        stage_frames = {stage: [] for stage in self.stages.keys()}
        frame_count = 0
        prev_frame_data = None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip = max(1, int(fps / 15))  # Process 15 frames per second
        
        print(f"Video info:")
        print(f"- FPS: {fps}")
        print(f"- Total frames: {total_frames}")
        print(f"- Processing every {frame_skip} frames")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            frame_count += 1
            
            # Skip frames if needed
            if frame_count % frame_skip != 0:
                continue
                
            # Extract pose data
            pose_data = self.extract_pose_data(frame)
            if pose_data:
                frames_data.append(pose_data)
                
                # Detect stages
                if not self.stages['stance']['detected'] and self.detect_stance(pose_data):
                    self.stages['stance']['detected'] = True
                    self.stages['stance']['frame'] = pose_data
                    stage_frames['stance'].append(pose_data)
                
                elif not self.stages['grip']['detected'] and self.detect_grip(pose_data):
                    self.stages['grip']['detected'] = True
                    self.stages['grip']['frame'] = pose_data
                    stage_frames['grip'].append(pose_data)
                
                elif not self.stages['bend']['detected'] and self.detect_bend(pose_data):
                    self.stages['bend']['detected'] = True
                    self.stages['bend']['frame'] = pose_data
                    stage_frames['bend'].append(pose_data)
                
                elif not self.stages['upward']['detected'] and prev_frame_data and self.detect_upward_motion(pose_data, prev_frame_data):
                    self.stages['upward']['detected'] = True
                    self.stages['upward']['frame'] = pose_data
                    stage_frames['upward'].append(pose_data)
                
                elif not self.stages['release']['detected'] and prev_frame_data and self.detect_release(pose_data, prev_frame_data):
                    self.stages['release']['detected'] = True
                    self.stages['release']['frame'] = pose_data
                    stage_frames['release'].append(pose_data)
                
                elif not self.stages['follow_through']['detected'] and self.detect_follow_through(pose_data):
                    self.stages['follow_through']['detected'] = True
                    self.stages['follow_through']['frame'] = pose_data
                    stage_frames['follow_through'].append(pose_data)
                
                elif not self.stages['landing']['detected'] and prev_frame_data and self.detect_landing(pose_data, prev_frame_data):
                    self.stages['landing']['detected'] = True
                    self.stages['landing']['frame'] = pose_data
                    stage_frames['landing'].append(pose_data)
                
                prev_frame_data = pose_data
            
        cap.release()
        
        # Print stage detection results
        print("\nStage Detection Results:")
        for stage, data in self.stages.items():
            print(f"- {stage}: {'Detected' if data['detected'] else 'Not detected'}")
        
        # Save the analyzed data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if is_reference:
            filename = f"{player_name}_{timestamp}.json"
            save_path = os.path.join(self.reference_dir, filename)
        else:
            filename = f"user_shot_{timestamp}.json"
            save_path = os.path.join(self.user_dir, filename)
            
        with open(save_path, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'player_name': player_name if is_reference else 'user',
                'frames': frames_data,
                'stages': stage_frames,
                'processing_info': {
                    'total_frames': frame_count,
                    'stages_detected': {stage: data['detected'] for stage, data in self.stages.items()}
                }
            }, f)
            
        return save_path

    def compare_shots(self, user_shot_path, reference_shot_path):
        """Compare a user's shot with a reference shot"""
        # Load both shots
        with open(user_shot_path, 'r') as f:
            user_data = json.load(f)
        with open(reference_shot_path, 'r') as f:
            reference_data = json.load(f)
            
        # TODO: Implement comparison logic
        # This will involve:
        # 1. Aligning the shots (finding similar poses)
        # 2. Comparing key angles and positions
        # 3. Generating feedback
        
        return {
            'similarity_score': 0.0,  # Placeholder
            'feedback': []  # Placeholder for feedback points
        }

# Example usage

if __name__ == "__main__":
    analyzer = PoseAnalyzer()
    # Example: analyzer.analyze_shot("path_to_video.mp4", is_reference=True, player_name="steph_curry") 