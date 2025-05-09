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
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        # Create directories if they don't exist
        self.data_dir = "shot_data"
        self.reference_dir = os.path.join(self.data_dir, "reference_shots")
        self.user_dir = os.path.join(self.data_dir, "user_shots")
        
        for directory in [self.data_dir, self.reference_dir, self.user_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

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

    def analyze_shot(self, video_path, is_reference=False, player_name=None):
        """Analyze a shooting video and save the pose data"""
        cap = cv2.VideoCapture(video_path)
        frames_data = []
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            pose_data = self.extract_pose_data(frame)
            if pose_data:
                frames_data.append(pose_data)
                
        cap.release()
        
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
                'frames': frames_data
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