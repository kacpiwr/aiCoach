import os
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class ShotAnalyzer:
    def __init__(self):
        self.data_dir = "shot_data/reference_shots"
        self.output_dir = "analysis_results"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
    def load_data(self):
        """Load all pose data from JSON files and extract release points"""
        all_data = []
        shot_labels = []
        
        # Find all analysis files
        files = [f for f in os.listdir(self.data_dir) 
                if f.endswith('.json')]
        
        for file in files:
            with open(os.path.join(self.data_dir, file), 'r') as f:
                data = json.load(f)
                
                if 'frames' not in data:
                    print(f"Warning: No frames found in {file}")
                    continue
                
                # Find the frame with the highest wrist position (release point)
                best_frame = None
                highest_wrist = -float('inf')
                
                for frame_data in data['frames']:
                    try:
                        # Get right wrist y-coordinate (index 16)
                        wrist_y = frame_data[16]['y']
                        if wrist_y > highest_wrist:
                            highest_wrist = wrist_y
                            best_frame = frame_data
                    except (IndexError, KeyError) as e:
                        print(f"Warning: Could not process frame in {file}: {str(e)}")
                        continue
                
                if best_frame is not None:
                    all_data.append(best_frame)
                    shot_labels.append(file)
        
        if not all_data:
            raise ValueError("No valid pose data found in any files")
            
        return all_data, np.array(shot_labels)
    
    def extract_features(self, pose_data):
        """Extract relevant features from pose data"""
        features = []
        
        # Handle single pose data
        if not isinstance(pose_data, list):
            pose_data = [pose_data]
        
        for frame_data in pose_data:
            try:
                # Extract key points for shooting form analysis using MediaPipe indices
                # Right side points
                right_shoulder = frame_data[12]  # Right shoulder
                right_elbow = frame_data[14]     # Right elbow
                right_wrist = frame_data[16]     # Right wrist
                right_hip = frame_data[24]       # Right hip
                right_knee = frame_data[26]      # Right knee
                right_ankle = frame_data[28]     # Right ankle
                
                # Calculate angles
                right_elbow_angle = self._calculate_angle(
                    right_shoulder, right_elbow, right_wrist)
                right_shoulder_angle = self._calculate_angle(
                    right_hip, right_shoulder, right_elbow)
                right_knee_angle = self._calculate_angle(
                    right_hip, right_knee, right_ankle)
                
                # Calculate positions
                wrist_height = right_wrist['y']
                elbow_height = right_elbow['y']
                shoulder_height = right_shoulder['y']
                
                # Calculate shot arc
                shot_arc = self._calculate_shot_arc(right_wrist, right_shoulder)
                
                # Calculate release point
                release_point = self._calculate_release_point(right_wrist, right_shoulder)
                
                # Combine features
                frame_features = [
                    right_elbow_angle,
                    right_shoulder_angle,
                    right_knee_angle,
                    wrist_height,
                    elbow_height,
                    shoulder_height,
                    right_wrist['x'] - right_shoulder['x'],  # Horizontal distance
                    right_wrist['z'] - right_shoulder['z'],  # Depth
                    shot_arc,
                    release_point['x'],
                    release_point['y'],
                    release_point['z']
                ]
                
                features.append(frame_features)
            except Exception as e:
                print(f"Warning: Could not extract features from frame: {str(e)}")
                continue
        
        if not features:
            raise ValueError("No valid features could be extracted from any frame")
            
        return np.array(features)
    
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
    
    def _calculate_shot_arc(self, wrist, shoulder):
        """Calculate the arc of the shot"""
        return np.arctan2(wrist['y'] - shoulder['y'], 
                         wrist['x'] - shoulder['x'])
    
    def _calculate_release_point(self, wrist, shoulder):
        """Calculate the release point of the shot"""
        return {
            'x': wrist['x'] - shoulder['x'],
            'y': wrist['y'] - shoulder['y'],
            'z': wrist['z'] - shoulder['z']
        }
    
    def analyze_shooting_patterns(self):
        """Analyze shooting patterns using machine learning"""
        # Load and preprocess data
        print("Loading pose data...")
        pose_data, labels = self.load_data()
        
        print("Extracting features...")
        features = self.extract_features(pose_data)
        
        # Create feature names
        feature_names = [
            'Elbow Angle', 'Shoulder Angle', 'Knee Angle',
            'Wrist Height', 'Elbow Height', 'Shoulder Height',
            'Horizontal Distance', 'Depth',
            'Shot Arc', 'Release X', 'Release Y', 'Release Z'
        ]
        
        # Calculate statistics for each feature
        feature_stats = {}
        for i, name in enumerate(feature_names):
            feature_stats[name] = {
                'mean': float(np.mean(features[:, i])),
                'std': float(np.std(features[:, i])),
                'min': float(np.min(features[:, i])),
                'max': float(np.max(features[:, i]))
            }
        
        # Save analysis results
        self._save_results(feature_stats, feature_names)
        
        # Print insights
        self._print_insights(feature_stats, feature_names)
        
        # Visualize results
        self._plot_results(features, labels, feature_stats, feature_names)
    
    def _plot_results(self, features, labels, feature_stats, feature_names):
        """Create visualizations of the analysis"""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Feature distributions
        plt.subplot(2, 1, 1)
        feature_means = [stats['mean'] for stats in feature_stats.values()]
        feature_stds = [stats['std'] for stats in feature_stats.values()]
        plt.barh(feature_names, feature_means, xerr=feature_stds)
        plt.title('Feature Distributions at Release Point')
        plt.xlabel('Value')
        
        # Plot 2: Feature correlations
        plt.subplot(2, 1, 2)
        correlation_matrix = np.corrcoef(features.T)
        sns.heatmap(correlation_matrix, 
                   xticklabels=feature_names,
                   yticklabels=feature_names,
                   cmap='coolwarm',
                   center=0)
        plt.title('Feature Correlations')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'steph_curry_analysis.png'))
        plt.close()
    
    def _print_insights(self, feature_stats, feature_names):
        """Print insights from the analysis"""
        print("\nSteph Curry Shooting Form Analysis Insights:")
        print("-" * 50)
        
        # Print key statistics for each feature
        for name, stats in feature_stats.items():
            print(f"\n{name}:")
            print(f"  Mean: {stats['mean']:.2f}")
            print(f"  Std:  {stats['std']:.2f}")
            print(f"  Range: {stats['min']:.2f} to {stats['max']:.2f}")
        
        print("\nKey Findings:")
        print("1. Most consistent features (lowest standard deviation):")
        sorted_features = sorted(feature_stats.items(), key=lambda x: x[1]['std'])
        for name, stats in sorted_features[:3]:
            print(f"   - {name}: {stats['std']:.2f}")
        
        print("\n2. Most variable features (highest standard deviation):")
        for name, stats in sorted_features[-3:]:
            print(f"   - {name}: {stats['std']:.2f}")
    
    def _save_results(self, feature_stats, feature_names):
        """Save analysis results to JSON file"""
        results = {
            'feature_statistics': feature_stats,
            'feature_names': feature_names,
            'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        output_file = os.path.join(self.output_dir, 'steph_curry_analysis.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nAnalysis results saved to: {output_file}")

def main():
    analyzer = ShotAnalyzer()
    analyzer.analyze_shooting_patterns()

if __name__ == "__main__":
    main() 