#!/usr/bin/env python3
import os
import sys
import argparse

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.analysis.analyze_user_shot import UserShotAnalyzer

def main():
    parser = argparse.ArgumentParser(description='Analyze a basketball shot against Steph Curry\'s form')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--user', '-u', required=True, help='Name of the user')
    parser.add_argument('--method', '-m', choices=['direct', 'ml'], default='ml',
                      help='Analysis method: direct (feature comparison) or ml (machine learning) [default: ml]')
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)
    
    try:
        analyzer = UserShotAnalyzer()
        results_path = analyzer.analyze_shot(args.video_path, args.user, args.method)
        print(f"\nAnalysis results saved to: {results_path}")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 