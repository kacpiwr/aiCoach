import os
import json
import numpy as np
import joblib
from datetime import datetime
import cv2
import mediapipe as mp
from .pose_analyzer import PoseAnalyzer
from .train_curry_model import extract_features_from_frame

class UserShotAnalyzer:
    def __init__(self):
        self.pose_analyzer = PoseAnalyzer()
        self.models_dir = "data/models"
        
        # Load the latest model and scaler
        self.model, self.scaler = self._load_latest_model()
        
        # Define feature mapping
        self.feature_mapping = {
            'elbow_angle': 'Elbow Angle',
            'shoulder_angle': 'Shoulder Angle',
            'knee_angle': 'Knee Angle',
            'wrist_height': 'Wrist Height',
            'elbow_height': 'Elbow Height',
            'horizontal_distance': 'Horizontal Distance',
            'depth': 'Depth',
            'shot_arc': 'Shot Arc',
            'release_x': 'Release X',
            'release_y': 'Release Y',
            'release_z': 'Release Z'
        }
        
    def _load_latest_model(self):
        """Load the most recent trained model and scaler"""
        model_files = [f for f in os.listdir(self.models_dir) if f.startswith('curry_shot_model_')]
        if not model_files:
            raise FileNotFoundError("No trained model found. Please train the model first.")
        
        # Get the latest model file
        latest_model = sorted(model_files)[-1]
        model_path = os.path.join(self.models_dir, latest_model)
        scaler_path = os.path.join(self.models_dir, latest_model.replace('model', 'scaler'))
        
        print(f"Loading model: {latest_model}")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        return model, scaler
    
    def _save_recommendations_txt(self, analysis, user_name, timestamp):
        """Save recommendations to a human-readable text file"""
        results_dir = "data/processed/analysis"
        txt_path = os.path.join(results_dir, f"{user_name}_{timestamp}_recommendations.txt")
        
        with open(txt_path, 'w') as f:
            f.write(f"Shot Form Analysis for {user_name}\n")
            f.write(f"Date: {analysis['timestamp']}\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Overall Similarity to Curry's Form: {analysis['similarity_score']:.1%}\n\n")
            
            # Write strengths (features with score > 80)
            f.write("STRENGTHS:\n")
            f.write("-"*20 + "\n")
            strengths = [(name, data) for name, data in analysis['feature_analysis'].items() 
                        if data['score'] > 80]
            for name, data in strengths:
                f.write(f"• {name}: {data['score']:.1f}/100 - ")
                diff = data['value'] - data['reference_mean']
                f.write(f"{'Very close' if abs(diff) < 0.1 else 'Similar'} to Curry's form ")
                f.write(f"(difference: {diff:.3f})\n")
            f.write("\n")
            
            # Write areas for improvement (features with score < 70)
            f.write("AREAS FOR IMPROVEMENT:\n")
            f.write("-"*20 + "\n")
            improvements = [(name, data) for name, data in analysis['feature_analysis'].items() 
                          if data['score'] < 70]
            for name, data in improvements:
                f.write(f"• {name}: {data['score']:.1f}/100 - ")
                if data['score'] < 60:
                    f.write("Significant difference from Curry's form\n")
                else:
                    f.write("Moderate difference from Curry's form\n")
            f.write("\n")
            
            # Write key differences
            f.write("KEY DIFFERENCES FROM CURRY'S FORM:\n")
            f.write("-"*20 + "\n")
            key_diffs = sorted(analysis['feature_analysis'].items(), 
                             key=lambda x: abs(x[1]['value'] - x[1]['reference_mean']), 
                             reverse=True)[:3]
            for name, data in key_diffs:
                diff = data['value'] - data['reference_mean']
                f.write(f"• {name}: {diff:.3f} units ")
                f.write(f"{'higher' if diff > 0 else 'lower'} than Curry's average of ")
                f.write(f"{data['reference_mean']:.3f}\n")
            f.write("\n")
            
            # Write recommendations
            f.write("RECOMMENDATIONS:\n")
            f.write("-"*20 + "\n")
            for rec in analysis['recommendations']:
                f.write(f"• Work on {rec['action']}ing {rec['feature'].lower()} - ")
                f.write(f"current: {rec['current']:.3f}, target: {rec['target']:.3f}\n")
            
        return txt_path
    
    def analyze_shot(self, video_path, user_name):
        """Analyze a user's shot and compare it to Curry's form"""
        print(f"\nAnalyzing shot for {user_name}...")
        print("="*50)
        
        # Process the video to get pose data
        print("Processing video...")
        pose_data_path = self.pose_analyzer.analyze_shot(
            video_path,
            is_reference=False,
            player_name=user_name
        )
        
        # Load the pose data
        with open(pose_data_path, 'r') as f:
            pose_data = json.load(f)
        
        # Find the best frame (highest wrist position)
        best_frame = None
        highest_wrist = -float('inf')
        
        for frame in pose_data['frames']:
            wrist = frame[16]  # Right wrist index
            if wrist['visibility'] > 0.5 and wrist['y'] > highest_wrist:
                highest_wrist = wrist['y']
                best_frame = frame
        
        if best_frame is None:
            raise ValueError("No valid pose data found in the video")
        
        # Extract features from the best frame
        features = extract_features_from_frame(best_frame)
        if features is None:
            raise ValueError("Could not extract features from the shot")
        
        # Remove visibility from features and map feature names
        features.pop('visibility', None)
        mapped_features = {self.feature_mapping[k]: v for k, v in features.items()}
        
        # Convert features to array and scale
        feature_array = np.array([list(mapped_features.values())])
        scaled_features = self.scaler.transform(feature_array)
        
        # Get model prediction (use decision_function for similarity score)
        try:
            similarity_score = self.model.decision_function(scaled_features)[0]
            # Convert to probability-like score between 0 and 1
            similarity_score = 1 / (1 + np.exp(-similarity_score))
        except:
            # Fallback to predict if decision_function is not available
            similarity_score = float(self.model.predict(scaled_features)[0])
        
        # Generate detailed analysis
        analysis = self._generate_analysis(mapped_features, similarity_score)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "data/processed/analysis"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save JSON results
        json_path = os.path.join(results_dir, f"{user_name}_{timestamp}_analysis.json")
        with open(json_path, 'w') as f:
            json.dump(analysis, f, indent=4)
        
        # Save text recommendations
        txt_path = self._save_recommendations_txt(analysis, user_name, timestamp)
        
        print("\nAnalysis complete!")
        print(f"Results saved to: {json_path}")
        print(f"Recommendations saved to: {txt_path}")
        print("\nSummary:")
        print(f"Overall similarity to Curry's form: {similarity_score:.1%}")
        print("\nTop recommendations:")
        for rec in analysis['recommendations'][:3]:
            print(f"- {rec['feature']}: {rec['action']} by {abs(rec['current'] - rec['target']):.2f}")
        
        return json_path, txt_path
    
    def _generate_analysis(self, features, similarity_score):
        """Generate detailed analysis of the shot"""
        # Load Curry's reference statistics
        with open('data/reference/reference_shots/steph_curry_analysis.json', 'r') as f:
            reference_stats = json.load(f)['feature_statistics']
        
        analysis = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'similarity_score': float(similarity_score),
            'feature_analysis': {},
            'recommendations': []
        }
        
        # Analyze each feature
        for feature, value in features.items():
            ref_stats = reference_stats[feature]
            z_score = abs(value - ref_stats['mean']) / ref_stats['std']
            
            # Calculate feature score (0-100)
            feature_score = 100 * np.exp(-0.5 * z_score)
            
            analysis['feature_analysis'][feature] = {
                'value': float(value),
                'reference_mean': float(ref_stats['mean']),
                'reference_std': float(ref_stats['std']),
                'score': float(feature_score),
                'z_score': float(z_score)
            }
            
            # Generate recommendations for significant differences
            if z_score > 1.5:
                direction = "decrease" if value > ref_stats['mean'] else "increase"
                analysis['recommendations'].append({
                    'feature': feature,
                    'action': direction,
                    'current': float(value),
                    'target': float(ref_stats['mean']),
                    'importance': 'high' if z_score > 2.0 else 'medium'
                })
        
        # Sort recommendations by importance
        analysis['recommendations'].sort(key=lambda x: x['importance'], reverse=True)
        
        return analysis

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze a basketball shot against Steph Curry\'s form')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('user_name', help='Name of the user')
    args = parser.parse_args()
    
    analyzer = UserShotAnalyzer()
    analyzer.analyze_shot(args.video_path, args.user_name)

if __name__ == "__main__":
    main() 