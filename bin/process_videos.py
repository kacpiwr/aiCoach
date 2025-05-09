import os
import sys
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.analysis.pose_analyzer import PoseAnalyzer
from src.analysis.shot_analyzer import ShotAnalyzer

def process_all_videos():
    analyzer = PoseAnalyzer()
    videos_dir = "data/raw/nba_videos/cut_shots"
    
    # Get all Steph Curry video files
    video_files = [f for f in os.listdir(videos_dir) if f.startswith('steph_curry_') and f.endswith('.mp4')]
    print(f"\nFound {len(video_files)} Steph Curry videos to process")
    print("Videos to process:")
    for video in video_files:
        print(f"- {video}")
    
    # Process each video
    for video_file in video_files:
        video_path = os.path.join(videos_dir, video_file)
        print(f"\n{'='*50}")
        print(f"Processing {video_file}...")
        print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
        
        # Analyze the video and save pose data
        analysis_path = analyzer.analyze_shot(
            video_path,
            is_reference=True,
            player_name="steph_curry"
        )
        
        print(f"Finished at: {datetime.now().strftime('%H:%M:%S')}")
        print(f"Analysis saved to: {analysis_path}")
        print(f"{'='*50}\n")

def analyze_test_shot():
    print("\nAnalyzing test shot (curryTest)...")
    print("="*50)
    
    # Initialize analyzers
    pose_analyzer = PoseAnalyzer()
    shot_analyzer = ShotAnalyzer()
    
    # Process the test shot
    test_video = "data/raw/users_shots/curryTest.mp4"
    print("\nStep 1: Processing test video...")
    test_analysis = pose_analyzer.analyze_shot(
        test_video,
        is_reference=False,
        player_name="test_curry"
    )
    print(f"Test shot analysis saved to: {test_analysis}")
    
    # Run the shot analysis to compare with reference data
    print("\nStep 2: Analyzing shooting patterns...")
    shot_analyzer.analyze_shooting_patterns()
    
    print("\nAnalysis complete! Check the analysis_results directory for detailed comparison.")
    print("="*50)

if __name__ == "__main__":
    analyze_test_shot()  # Run test analysis
    # process_all_videos() 