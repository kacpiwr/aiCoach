import os
import sys
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.analysis.pose_analyzer import PoseAnalyzer

def rescan_reference_shots():
    print("\nRescanning all Curry's shots with stage detection...")
    print("="*50)
    
    analyzer = PoseAnalyzer()
    videos_dir = "data/raw/nba_videos/cut_shots"
    
    # Get all video files
    video_files = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]
    print(f"\nFound {len(video_files)} videos to process")
    print("Videos to process:")
    for video in video_files:
        print(f"- {video}")
    
    # Process each video
    successful_scans = 0
    for video_file in video_files:
        video_path = os.path.join(videos_dir, video_file)
        print(f"\n{'='*50}")
        print(f"Processing {video_file}...")
        print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            # All videos are Curry's shots
            analysis_path = analyzer.analyze_shot(
                video_path,
                is_reference=True,
                player_name="steph_curry"
            )
            
            print(f"Finished at: {datetime.now().strftime('%H:%M:%S')}")
            print(f"Analysis saved to: {analysis_path}")
            successful_scans += 1
            
        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")
        
        print(f"{'='*50}\n")
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed {successful_scans} out of {len(video_files)} videos")
    print("="*50)

if __name__ == "__main__":
    rescan_reference_shots() 