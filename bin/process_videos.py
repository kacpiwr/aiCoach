import os
from pose_analyzer import PoseAnalyzer

def process_all_videos():
    analyzer = PoseAnalyzer()
    videos_dir = "nba_videos/cut_shots"
    
    # Get all video files
    video_files = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]
    print(f"Found {len(video_files)} videos to process")
    
    # Process each video
    for video_file in video_files:
        video_path = os.path.join(videos_dir, video_file)
        print(f"\nProcessing {video_file}...")
        
        # Analyze the video and save pose data
        analysis_path = analyzer.analyze_shot(
            video_path,
            is_reference=True,
            player_name="steph_curry"
        )
        print(f"Analysis saved to: {analysis_path}")

if __name__ == "__main__":
    process_all_videos() 