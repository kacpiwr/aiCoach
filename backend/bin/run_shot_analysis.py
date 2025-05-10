import os
import json
from datetime import datetime
import sys

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis.pose_analyzer import PoseAnalyzer
from analysis.shot_comparator import ShotComparator
from analysis.analyze_findings import FindingsAnalyzer

class ShotAnalysisPipeline:
    def __init__(self):
        # Initialize directories
        self.directories = {
            'videos': 'data/raw/users_shots',
            'pose_data': 'data/processed/shot_data',
            'comparison': 'data/results/comparison_results',
            'findings': 'data/results/analysis_findings'
        }
        
        # Create directories if they don't exist
        for directory in self.directories.values():
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Initialize analyzers
        self.pose_analyzer = PoseAnalyzer()
        self.shot_comparator = ShotComparator()
        self.findings_analyzer = FindingsAnalyzer()
    
    def process_video(self, video_path, player_name):
        """Process a video file and extract pose data"""
        print(f"\nProcessing video: {video_path}")
        print("-" * 50)
        
        # Analyze the shot
        analysis_path = self.pose_analyzer.analyze_shot(
            video_path,
            is_reference=False,
            player_name=player_name
        )
        
        print(f"Pose analysis saved to: {analysis_path}")
        return analysis_path
    
    def compare_with_curry(self, pose_data_path, player_name):
        """Compare the shot with Curry's form"""
        print(f"\nComparing {player_name}'s shot with Steph Curry's form...")
        print("-" * 50)
        
        # Load the pose data
        with open(pose_data_path, 'r') as f:
            pose_data = json.load(f)
        
        # Get the frame with the best shot form (highest wrist position)
        best_frame = None
        highest_wrist = -float('inf')
        
        for frame in pose_data['frames']:
            wrist_y = frame[16]['y']
            if wrist_y > highest_wrist:
                highest_wrist = wrist_y
                best_frame = frame
        
        if best_frame is None:
            raise ValueError("No valid pose data found in the video")
        
        # Compare with Curry's form
        json_path, vis_path = self.shot_comparator.save_comparison(best_frame, player_name)
        
        print(f"Comparison results saved to: {json_path}")
        print(f"Visualization saved to: {vis_path}")
        return json_path
    
    def analyze_findings(self, comparison_path):
        """Analyze the comparison results and generate findings"""
        print("\nAnalyzing findings...")
        print("-" * 50)
        
        # Analyze findings
        findings = self.findings_analyzer.analyze_comparison(comparison_path)
        
        # Save findings in both formats
        base_path = self.findings_analyzer.save_findings(findings)
        
        print(f"Findings saved to:")
        print(f"- {base_path}_findings.txt (human-readable)")
        print(f"- {base_path}_findings.json (machine-readable)")
        
        return base_path
    
    def run_analysis(self, video_path, player_name):
        """Run the complete analysis pipeline"""
        try:
            # Step 1: Process video and extract pose data
            pose_data_path = self.process_video(video_path, player_name)
            
            # Step 2: Compare with Curry's form
            comparison_path = self.compare_with_curry(pose_data_path, player_name)
            
            # Step 3: Analyze findings
            findings_path = self.analyze_findings(comparison_path)
            
            print("\nAnalysis pipeline completed successfully!")
            print("=" * 50)
            print(f"All results have been saved in their respective directories:")
            print(f"- Pose data: {pose_data_path}")
            print(f"- Comparison: {comparison_path}")
            print(f"- Findings: {findings_path}_findings.txt")
            
            return {
                'pose_data': pose_data_path,
                'comparison': comparison_path,
                'findings': f"{findings_path}_findings.txt"
            }
            
        except Exception as e:
            print(f"\nError during analysis: {str(e)}")
            return None

def main():
    # Initialize the pipeline
    pipeline = ShotAnalysisPipeline()
    
    # Get the most recent video from users_shots directory
    video_dir = "data/raw/users_shots"
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    
    if not video_files:
        print("No video files found in users_shots directory!")
        return
    
    # Get most recent video
    latest_video = max(video_files, 
                      key=lambda x: os.path.getctime(os.path.join(video_dir, x)))
    video_path = os.path.join(video_dir, latest_video)
    
    # Extract player name from video filename (remove extension)
    player_name = os.path.splitext(latest_video)[0]
    
    print(f"Starting analysis for video: {latest_video}")
    print("=" * 50)
    
    # Run the analysis pipeline
    results = pipeline.run_analysis(video_path, player_name)
    
    if results:
        print("\nAnalysis Summary:")
        print("=" * 50)
        print(f"Player: {player_name}")
        print(f"Video: {latest_video}")
        print("\nResults saved in:")
        for key, path in results.items():
            print(f"- {key}: {path}")

if __name__ == "__main__":
    main() 