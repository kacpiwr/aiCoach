import os
import cv2
import yt_dlp
import numpy as np
from pose_analyzer import PoseAnalyzer
from datetime import datetime
import json

class VideoCollector:
    def __init__(self):
        self.analyzer = PoseAnalyzer()
        self.videos_dir = "nba_videos"
        if not os.path.exists(self.videos_dir):
            os.makedirs(self.videos_dir)

    def download_video(self, url, player_name):
        """Download a video from YouTube URL"""
        ydl_opts = {
            'format': 'best[height<=720]',  # Limit to 720p for processing speed
            'outtmpl': os.path.join(self.videos_dir, f'{player_name}_%(id)s.%(ext)s'),
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
                return True
        except Exception as e:
            print(f"Error downloading video: {e}")
            return False

    def analyze_video(self, video_path, player_name):
        """Analyze the entire video and save pose data"""
        cap = cv2.VideoCapture(video_path)
        frames_data = []
        frame_count = 0
        
        print(f"Analyzing video: {video_path}")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            # Process frame with pose detection
            pose_data = self.analyzer.extract_pose_data(frame)
            if pose_data:
                frames_data.append(pose_data)
            
            frame_count += 1
            if frame_count % 30 == 0:  # Print progress every 30 frames
                print(f"Processed {frame_count} frames...")
                
        cap.release()
        
        # Save the analyzed data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.videos_dir, f"{player_name}_analysis_{timestamp}.json")
        
        with open(output_path, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'player_name': player_name,
                'total_frames': frame_count,
                'frames_with_pose': len(frames_data),
                'frames': frames_data
            }, f)
            
        print(f"Analysis complete. Processed {frame_count} frames, found pose in {len(frames_data)} frames")
        return output_path

def main():
    collector = VideoCollector()
    
    # NBA shooting videos
    videos = [
        {
            'url': 'https://www.youtube.com/watch?v=nXCol8YeqxQ',  # Steph Curry shooting compilation
            'player': 'steph_curry'
        },
        {
            'url': 'https://www.youtube.com/watch?v=_uSVVxBJnz0',  # Klay Thompson shooting compilation
            'player': 'klay_thompson'
        }
    ]
    
    for video in videos:
        print(f"\nProcessing {video['player']}'s video...")
        if collector.download_video(video['url'], video['player']):
            # Find the downloaded video file
            video_files = [f for f in os.listdir(collector.videos_dir) 
                         if f.startswith(video['player']) and f.endswith('.mp4')]
            if video_files:
                video_path = os.path.join(collector.videos_dir, video_files[-1])
                analysis_path = collector.analyze_video(video_path, video['player'])
                print(f"Analysis saved to: {analysis_path}")
            else:
                print(f"Could not find downloaded video for {video['player']}")

if __name__ == "__main__":
    main() 