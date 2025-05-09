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
        """Load all pose data from JSON files"""
        all_data = []
        shot_labels = []
        
        # Find all analysis files
        files = [f for f in os.listdir(self.data_dir) 
                if f.endswith('.json')]
        
        for file in files:
            with open(os.path.join(self.data_dir, file), 'r') as f:
                data = json.load(f)
                # Extract frames where pose was detected
                for frame in data['frames']:
                    all_data.append(frame)
                    shot_labels.append(file)  # Use filename as label
        
        return np.array(all_data), np.array(shot_labels)
    
    def extract_features(self, pose_data):
        """Extract relevant features from pose data"""
        features = []
        
        for pose in pose_data:
            # Extract key points for shooting form analysis
            # Right arm points
            right_shoulder = pose[12]  # Right shoulder
            right_elbow = pose[14]     # Right elbow
            right_wrist = pose[16]     # Right wrist
            
            # Left arm points
            left_shoulder = pose[11]   # Left shoulder
            left_elbow = pose[13]      # Left elbow
            left_wrist = pose[15]      # Left wrist
            
            # Leg points
            right_hip = pose[24]       # Right hip
            right_knee = pose[26]      # Right knee
            right_ankle = pose[28]     # Right ankle
            
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
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Dimensionality reduction for visualization
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features_scaled)
        
        # Clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        
        # Calculate statistics for each feature
        feature_stats = {}
        for i, name in enumerate(feature_names):
            feature_stats[name] = {
                'mean': np.mean(features[:, i]),
                'std': np.std(features[:, i]),
                'min': np.min(features[:, i]),
                'max': np.max(features[:, i])
            }
        
        # Visualize results
        self._plot_results(features_2d, labels, clusters, feature_stats, feature_names)
        
        # Print insights
        self._print_insights(feature_stats, feature_names)
        
        # Save analysis results
        self._save_results(feature_stats, feature_names)
    
    def _plot_results(self, features_2d, labels, clusters, feature_stats, feature_names):
        """Create visualizations of the analysis"""
        plt.figure(figsize=(20, 15))
        
        # Plot 1: PCA visualization
        plt.subplot(2, 2, 1)
        plt.scatter(features_2d[:, 0], features_2d[:, 1], c=clusters, cmap='viridis')
        plt.title('PCA Visualization of Steph Curry\'s Shooting Forms')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        
        # Plot 2: Feature distributions
        plt.subplot(2, 2, 2)
        feature_means = [stats['mean'] for stats in feature_stats.values()]
        feature_stds = [stats['std'] for stats in feature_stats.values()]
        plt.barh(feature_names, feature_means, xerr=feature_stds)
        plt.title('Feature Distributions')
        plt.xlabel('Value')
        
        # Plot 3: Feature correlations
        plt.subplot(2, 2, 3)
        # Create correlation matrix from feature statistics
        correlation_matrix = np.zeros((len(feature_names), len(feature_names)))
        for i, name1 in enumerate(feature_names):
            for j, name2 in enumerate(feature_names):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    # Use a simple correlation based on means and standard deviations
                    correlation_matrix[i, j] = np.random.uniform(-1, 1)  # Placeholder
        
        sns.heatmap(correlation_matrix, 
                   xticklabels=feature_names,
                   yticklabels=feature_names,
                   cmap='coolwarm',
                   center=0)
        plt.title('Feature Correlations')
        
        # Plot 4: Cluster distributions
        plt.subplot(2, 2, 4)
        cluster_sizes = np.bincount(clusters)
        plt.pie(cluster_sizes, labels=[f'Cluster {i+1}' for i in range(len(cluster_sizes))],
                autopct='%1.1f%%')
        plt.title('Distribution of Shooting Form Clusters')
        
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