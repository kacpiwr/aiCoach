import os
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime

class ShotComparator:
    def __init__(self):
        self.reference_dir = "analysis_results"
        self.output_dir = "comparison_results"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Load Curry's reference data
        self.reference_data = self._load_reference_data()
        
    def _load_reference_data(self):
        """Load Curry's reference shooting form data"""
        reference_file = os.path.join(self.reference_dir, "steph_curry_analysis.json")
        if not os.path.exists(reference_file):
            raise FileNotFoundError("Reference data not found. Run shot_analyzer.py first.")
            
        with open(reference_file, 'r') as f:
            return json.load(f)
    
    def _extract_features(self, pose_data):
        """Extract features from pose data in the same way as the analyzer"""
        # Extract key points
        right_shoulder = pose_data[12]
        right_elbow = pose_data[14]
        right_wrist = pose_data[16]
        right_hip = pose_data[24]
        right_knee = pose_data[26]
        right_ankle = pose_data[28]
        
        # Calculate angles
        elbow_angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
        shoulder_angle = self._calculate_angle(right_hip, right_shoulder, right_elbow)
        knee_angle = self._calculate_angle(right_hip, right_knee, right_ankle)
        
        # Calculate positions and metrics
        wrist_height = right_wrist['y']
        elbow_height = right_elbow['y']
        shoulder_height = right_shoulder['y']
        horizontal_distance = right_wrist['x'] - right_shoulder['x']
        depth = right_wrist['z'] - right_shoulder['z']
        
        # Calculate shot arc and release point
        shot_arc = np.arctan2(right_wrist['y'] - right_shoulder['y'], 
                             right_wrist['x'] - right_shoulder['x'])
        
        release_point = {
            'x': right_wrist['x'] - right_shoulder['x'],
            'y': right_wrist['y'] - right_shoulder['y'],
            'z': right_wrist['z'] - right_shoulder['z']
        }
        
        return {
            'Elbow Angle': elbow_angle,
            'Shoulder Angle': shoulder_angle,
            'Knee Angle': knee_angle,
            'Wrist Height': wrist_height,
            'Elbow Height': elbow_height,
            'Shoulder Height': shoulder_height,
            'Horizontal Distance': horizontal_distance,
            'Depth': depth,
            'Shot Arc': shot_arc,
            'Release X': release_point['x'],
            'Release Y': release_point['y'],
            'Release Z': release_point['z']
        }
    
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
    
    def compare_shot(self, user_pose_data):
        """Compare a user's shot to Curry's reference form"""
        # Extract features from user's shot
        user_features = self._extract_features(user_pose_data)
        
        # Get reference statistics
        ref_stats = self.reference_data['feature_statistics']
        
        # Calculate similarity scores and differences
        comparison = {
            'overall_score': 0,
            'feature_scores': {},
            'recommendations': []
        }
        
        total_score = 0
        feature_weights = {
            'Elbow Angle': 1.5,      # Important for shot power
            'Shoulder Angle': 1.2,    # Important for shot arc
            'Knee Angle': 1.0,        # Important for power generation
            'Release X': 1.8,         # Critical for accuracy
            'Release Y': 1.5,         # Critical for arc
            'Shot Arc': 1.3,          # Important for trajectory
            'Horizontal Distance': 1.2,# Important for consistency
            'Wrist Height': 1.1,      # Important for release point
            'Elbow Height': 1.0,      # Part of overall form
            'Shoulder Height': 1.0,   # Part of overall form
            'Depth': 1.0,             # Part of overall form
            'Release Z': 1.0          # Part of overall form
        }
        
        total_weight = sum(feature_weights.values())
        
        for feature, value in user_features.items():
            ref_mean = ref_stats[feature]['mean']
            ref_std = ref_stats[feature]['std']
            ref_range = ref_stats[feature]['max'] - ref_stats[feature]['min']
            
            # Calculate z-score
            z_score = abs(value - ref_mean) / (ref_std if ref_std > 0 else 1)
            
            # Convert to a 0-100 score (closer to 0 is better)
            score = 100 * np.exp(-0.5 * z_score)
            
            # Apply feature weight
            weighted_score = score * feature_weights[feature]
            total_score += weighted_score
            
            comparison['feature_scores'][feature] = {
                'score': score,
                'difference': value - ref_mean,
                'user_value': value,
                'reference_mean': ref_mean,
                'reference_std': ref_std
            }
            
            # Generate recommendations
            if z_score > 1.5:  # If more than 1.5 standard deviations away
                direction = "increase" if value < ref_mean else "decrease"
                recommendation = {
                    'feature': feature,
                    'action': direction,
                    'current': value,
                    'target': ref_mean,
                    'importance': 'high' if feature_weights[feature] > 1.2 else 'medium'
                }
                comparison['recommendations'].append(recommendation)
        
        # Calculate overall score (0-100)
        comparison['overall_score'] = total_score / total_weight
        
        # Sort recommendations by importance
        comparison['recommendations'].sort(key=lambda x: feature_weights[x['feature']], reverse=True)
        
        return comparison
    
    def visualize_comparison(self, comparison, output_path):
        """Create visualization of the shot comparison"""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Feature Scores
        plt.subplot(2, 1, 1)
        features = list(comparison['feature_scores'].keys())
        scores = [comparison['feature_scores'][f]['score'] for f in features]
        
        plt.barh(features, scores)
        plt.axvline(x=80, color='g', linestyle='--', label='Excellent (80+)')
        plt.axvline(x=60, color='y', linestyle='--', label='Good (60+)')
        plt.axvline(x=40, color='r', linestyle='--', label='Needs Work (<60)')
        plt.title(f'Shot Form Comparison (Overall Score: {comparison["overall_score"]:.1f})')
        plt.xlabel('Score (0-100)')
        plt.legend()
        
        # Plot 2: Differences from Reference
        plt.subplot(2, 1, 2)
        differences = [comparison['feature_scores'][f]['difference'] for f in features]
        colors = ['r' if d < 0 else 'g' for d in differences]
        
        plt.barh(features, differences, color=colors)
        plt.axvline(x=0, color='black', linestyle='-')
        plt.title('Differences from Curry\'s Form')
        plt.xlabel('Difference from Reference')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def save_comparison(self, user_pose_data, user_name):
        """Compare and save results of a user's shot"""
        comparison = self.compare_shot(user_pose_data)
        
        # Create timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save visualization
        vis_path = os.path.join(self.output_dir, f"{user_name}_{timestamp}_comparison.png")
        self.visualize_comparison(comparison, vis_path)
        
        # Save detailed results
        results = {
            'timestamp': timestamp,
            'user_name': user_name,
            'comparison': comparison
        }
        
        json_path = os.path.join(self.output_dir, f"{user_name}_{timestamp}_results.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Print summary
        print(f"\nShot Comparison Results for {user_name}")
        print("-" * 50)
        print(f"Overall Score: {comparison['overall_score']:.1f}/100")
        print("\nTop Recommendations:")
        for rec in comparison['recommendations'][:3]:
            print(f"- {rec['feature']}: {rec['action']} by {abs(rec['current'] - rec['target']):.2f}")
        
        print(f"\nDetailed results saved to: {json_path}")
        print(f"Visualization saved to: {vis_path}")
        
        return json_path, vis_path

def main():
    # Example usage
    comparator = ShotComparator()
    
    # Example pose data (this would come from your video processing)
    example_pose = {
        # This would be filled with actual pose data from the video
    }
    
    # Compare and save results
    json_path, vis_path = comparator.save_comparison(example_pose, "example_user")

if __name__ == "__main__":
    main() 