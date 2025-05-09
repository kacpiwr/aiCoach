import os
from pose_analyzer import PoseAnalyzer
from shot_comparator import ShotComparator
import json
import numpy as np

def process_and_compare_shot():
    # Initialize analyzers
    pose_analyzer = PoseAnalyzer()
    shot_comparator = ShotComparator()
    
    # Process the user's video
    video_path = "users_shots/User1.mp4"
    print(f"Processing video: {video_path}")
    
    # Analyze the shot
    analysis_path = pose_analyzer.analyze_shot(
        video_path,
        is_reference=False,
        player_name="anthony_edwards"
    )
    
    # Load the pose data
    with open(analysis_path, 'r') as f:
        pose_data = json.load(f)
    
    # Get the frame with the best shot form (highest wrist position)
    best_frame = None
    highest_wrist = -float('inf')
    
    for frame in pose_data['frames']:
        # Get wrist position (index 16 is right wrist)
        wrist_y = frame[16]['y']
        if wrist_y > highest_wrist:
            highest_wrist = wrist_y
            best_frame = frame
    
    if best_frame is None:
        print("No valid pose data found in the video")
        return
    
    # Compare with Curry's form
    print("\nComparing shot with Steph Curry's form...")
    json_path, vis_path = shot_comparator.save_comparison(best_frame, "anthony_edwards")
    
    print("\nAnalysis complete!")
    print(f"Results saved to: {json_path}")
    print(f"Visualization saved to: {vis_path}")

if __name__ == "__main__":
    process_and_compare_shot() 