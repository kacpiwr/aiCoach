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
        # Use absolute paths from environment variables
        self.models_dir = os.getenv("MODELS_DIR", "data/models")
        self.reference_dir = os.getenv("REFERENCE_DIR", "data/reference/reference_shots")
        self.processed_dir = os.getenv("PROCESSED_DIR", "data/processed")
        self.results_dir = os.getenv("RESULTS_DIR", "data/results/analysis")
        
        # Load the latest model and scaler
        self.model, self.scaler = self._load_latest_model()
        
        # Define feature mapping
        self.feature_mapping = {
            'elbow_angle': 'elbow_angle',
            'shoulder_angle': 'shoulder_angle',
            'knee_angle': 'knee_angle',
            'wrist_height': 'wrist_height',
            'elbow_height': 'elbow_height',
            'depth': 'depth',
            'shot_arc': 'shot_arc',
            'release_y': 'release_y'
        }
        
    def _load_latest_model(self):
        """Load the most recent trained model and scaler"""
        model_files = [f for f in os.listdir(self.models_dir) if f.startswith('curry_shot_model_')]
        if not model_files:
            raise FileNotFoundError(f"No trained model found in {self.models_dir}. Please train the model first.")
        
        # Get the latest model file
        latest_model = sorted(model_files)[-1]
        model_path = os.path.join(self.models_dir, latest_model)
        scaler_path = os.path.join(self.models_dir, latest_model.replace('model', 'scaler'))
        
        print(f"Loading model: {latest_model}")
        print(f"Model path: {model_path}")
        print(f"Scaler path: {scaler_path}")
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        return model, scaler
    
    def _save_recommendations_txt(self, stage_analyses, user_name, timestamp):
        """Save recommendations to a human-readable text file"""
        results_dir = os.path.join(self.results_dir, "analysis")
        os.makedirs(results_dir, exist_ok=True)
        txt_path = os.path.join(results_dir, f"{user_name}_{timestamp}_recommendations.txt")
        
        with open(txt_path, 'w') as f:
            f.write(f"Shot Form Analysis for {user_name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*50 + "\n\n")
            
            overall_similarity = sum(analysis['similarity_score'] for analysis in stage_analyses.values()) / len(stage_analyses)
            f.write(f"Overall Similarity to Curry's Form: {overall_similarity:.1f}%\n\n")
            
            # Write stage-by-stage analysis
            f.write("STAGE-BY-STAGE ANALYSIS:\n")
            f.write("-"*20 + "\n")
            for stage, analysis in stage_analyses.items():
                f.write(f"\n{stage.upper()} STAGE:\n")
                f.write(f"Similarity: {analysis['similarity_score']:.1f}%\n")
                
                # Write strengths (features with score > 80)
                f.write("Strengths:\n")
                strengths = [(name, data) for name, data in analysis['feature_analysis'].items() 
                            if data['score'] > 80]
                for name, data in strengths:
                    f.write(f"• {name}: {data['score']:.1f}/100 - ")
                    diff = data['value'] - data['reference_mean']
                    f.write(f"{'Very close' if abs(diff) < 0.1 else 'Similar'} to Curry's form ")
                    f.write(f"(difference: {diff:.3f})\n")
                
                # Write areas for improvement (features with score < 70)
                f.write("Areas for Improvement:\n")
                improvements = [(name, data) for name, data in analysis['feature_analysis'].items() 
                              if data['score'] < 70]
                for name, data in improvements:
                    f.write(f"• {name}: {data['score']:.1f}/100 - ")
                    if data['score'] < 60:
                        f.write("Significant difference from Curry's form\n")
                    else:
                        f.write("Moderate difference from Curry's form\n")
                
                # Write key differences
                f.write("Key Differences from Curry's Form:\n")
                key_diffs = sorted(analysis['feature_analysis'].items(), 
                                 key=lambda x: abs(x[1]['value'] - x[1]['reference_mean']), 
                                 reverse=True)[:3]
                for name, data in key_diffs:
                    diff = data['value'] - data['reference_mean']
                    f.write(f"• {name}: {diff:.3f} units ")
                    f.write(f"{'higher' if diff > 0 else 'lower'} than Curry's average of ")
                    f.write(f"{data['reference_mean']:.3f}\n")
            
            # Write detailed recommendations
            f.write("\nDETAILED RECOMMENDATIONS:\n")
            f.write("-"*20 + "\n")
            
            # Group recommendations by category
            lower_body = []
            upper_body = []
            positioning = []
            
            for stage, analysis in stage_analyses.items():
                for rec in analysis['recommendations']:
                    feature = rec['feature'].lower()
                    if feature in ['knee_angle', 'depth']:
                        lower_body.append(rec)
                    elif feature in ['elbow_angle', 'shoulder_angle', 'wrist_height', 'elbow_height', 'shot_arc']:
                        upper_body.append(rec)
                    else:
                        positioning.append(rec)
            
            # Write lower body recommendations
            if lower_body:
                f.write("1. Lower Body Mechanics:\n")
                for rec in lower_body:
                    if rec['feature'] == 'knee_angle':
                        f.write("   • Bend your knees more deeply during the shot preparation\n")
                        f.write("   • Aim for a 90-degree knee bend at the lowest point\n")
                        f.write("   • Keep your feet shoulder-width apart for better stability\n")
                    elif rec['feature'] == 'depth':
                        f.write("   • Maintain consistent depth in your shot preparation\n")
                        f.write("   • Practice shooting from different spots to improve spatial awareness\n")
                f.write("\n")
            
            # Write upper body recommendations
            if upper_body:
                f.write("2. Shot Release and Follow-through:\n")
                for rec in upper_body:
                    if rec['feature'] == 'elbow_angle':
                        f.write("   • Keep your elbow closer to your body (currently too high)\n")
                        f.write("   • Focus on a stronger wrist flick at release\n")
                    elif rec['feature'] == 'shoulder_angle':
                        f.write("   • Ensure your shooting hand follows through straight toward the basket\n")
                    elif rec['feature'] == 'shot_arc':
                        f.write("   • Practice one-handed form shots from close range to perfect the release\n")
                f.write("\n")
            
            # Write positioning recommendations
            if positioning:
                f.write("3. Body Positioning:\n")
                for rec in positioning:
                    if rec['feature'] == 'depth':
                        f.write("   • Maintain consistent depth in your shot preparation\n")
                        f.write("   • Practice shooting from different spots to improve spatial awareness\n")
                f.write("\n")
            
            # Write practice drills
            f.write("PRACTICE DRILLS:\n")
            f.write("-"*20 + "\n")
            
            f.write("1. Form Shooting Drill:\n")
            f.write("   • Start 2-3 feet from the basket\n")
            f.write("   • Focus on perfect form with deep knee bend\n")
            f.write("   • Shoot 50 shots with emphasis on proper mechanics\n")
            f.write("   • Gradually move back as form improves\n\n")
            
            f.write("2. Elbow Control Drill:\n")
            f.write("   • Stand in front of a mirror\n")
            f.write("   • Practice bringing the ball up with elbow tucked in\n")
            f.write("   • Focus on keeping elbow at 90 degrees\n")
            f.write("   • Repeat 20 times before shooting\n\n")
            
            f.write("3. Depth Control Drill:\n")
            f.write("   • Place markers at different distances\n")
            f.write("   • Practice shooting from each marker\n")
            f.write("   • Focus on maintaining consistent form\n")
            f.write("   • Track your success rate at each distance\n\n")
            
            # Write progress tracking
            f.write("PROGRESS TRACKING:\n")
            f.write("-"*20 + "\n")
            f.write("• Record your shooting percentage from each distance\n")
            f.write("• Note improvements in knee bend and elbow position\n")
            f.write("• Track consistency of release point\n")
            f.write("• Monitor overall shooting percentage\n\n")
            
            f.write("Remember: Focus on one aspect at a time. Master the fundamentals before moving to more advanced techniques.\n")
            
        return txt_path
    
    def _find_stage_frame(self, pose_data, stage_name):
        """Find the most appropriate frame for a given stage based on pose characteristics."""
        frames = pose_data['frames']
        
        if stage_name == 'stance':
            # Find the frame where the player is most stable (minimal movement)
            min_movement = float('inf')
            stance_frame = None
            for i in range(len(frames) - 1):
                current = frames[i]
                next_frame = frames[i + 1]
                # Calculate movement between frames using key points
                movement = sum(
                    abs(current[j]['x'] - next_frame[j]['x']) + 
                    abs(current[j]['y'] - next_frame[j]['y'])
                    for j in [0, 11, 12, 23, 24]  # Key points for stability
                    if current[j]['visibility'] > 0.5 and next_frame[j]['visibility'] > 0.5
                )
                if movement < min_movement:
                    min_movement = movement
                    stance_frame = current
            return stance_frame

        elif stage_name == 'bend':
            # Find the frame with maximum knee bend before release
            max_bend = float('-inf')
            bend_frame = None
            release_frame_idx = None
            
            # First find the release frame to ensure we only look at frames before release
            for i, frame in enumerate(frames):
                wrist = frame[16]  # Right wrist
                shoulder = frame[12]  # Right shoulder
                if wrist['visibility'] > 0.5 and shoulder['visibility'] > 0.5:
                    if wrist['y'] < shoulder['y']:  # Wrist above shoulder indicates release
                        release_frame_idx = i
                        break
            
            # If we found a release frame, only look at frames before it
            search_frames = frames[:release_frame_idx] if release_frame_idx is not None else frames
            
            for frame in search_frames:
                # Calculate knee angles for both legs
                left_knee = self._calculate_angle(
                    frame[23],  # Left hip
                    frame[25],  # Left knee
                    frame[27]   # Left ankle
                )
                right_knee = self._calculate_angle(
                    frame[24],  # Right hip
                    frame[26],  # Right knee
                    frame[28]   # Right ankle
                )
                
                # Also check hip angles to ensure proper shooting stance
                left_hip = self._calculate_angle(
                    frame[11],  # Left shoulder
                    frame[23],  # Left hip
                    frame[25]   # Left knee
                )
                right_hip = self._calculate_angle(
                    frame[12],  # Right shoulder
                    frame[24],  # Right hip
                    frame[26]   # Right knee
                )
                
                if all(angle is not None for angle in [left_knee, right_knee, left_hip, right_hip]):
                    # Calculate average knee bend (lower angle means more bend)
                    avg_knee_bend = (left_knee + right_knee) / 2
                    
                    # Calculate average hip angle
                    avg_hip_angle = (left_hip + right_hip) / 2
                    
                    # Check if this is a valid shooting stance
                    # - Knee angle should be between 90 and 150 degrees (typical shooting range)
                    # - Hip angle should be between 60 and 120 degrees (typical shooting range)
                    if 90 <= avg_knee_bend <= 150 and 60 <= avg_hip_angle <= 120:
                        # We want the frame with the lowest knee angle (most bend)
                        if avg_knee_bend < max_bend:
                            max_bend = avg_knee_bend
                            bend_frame = frame
            
            return bend_frame

        elif stage_name == 'release':
            # Find the frame with highest wrist position and appropriate elbow angle
            best_frame = None
            highest_wrist = float('-inf')
            for frame in frames:
                wrist = frame[16]  # Right wrist
                elbow = frame[14]  # Right elbow
                shoulder = frame[12]  # Right shoulder
                
                if wrist['visibility'] > 0.5 and elbow['visibility'] > 0.5 and shoulder['visibility'] > 0.5:
                    # Check if wrist is above shoulder (indicating release)
                    if wrist['y'] < shoulder['y']:
                        # Calculate elbow angle to ensure proper form
                        elbow_angle = self._calculate_angle(shoulder, elbow, wrist)
                        if elbow_angle is not None and 80 < elbow_angle < 160:  # Typical release angle range
                            if wrist['y'] > highest_wrist:
                                highest_wrist = wrist['y']
                                best_frame = frame
            return best_frame

        elif stage_name == 'follow_through':
            # Find the frame after release where the arm is fully extended
            release_frame_idx = None
            for i, frame in enumerate(frames):
                wrist = frame[16]  # Right wrist
                elbow = frame[14]  # Right elbow
                shoulder = frame[12]  # Right shoulder
                
                if wrist['visibility'] > 0.5 and elbow['visibility'] > 0.5 and shoulder['visibility'] > 0.5:
                    if wrist['y'] < shoulder['y']:  # Wrist above shoulder indicates release
                        release_frame_idx = i
                        break
            
            if release_frame_idx is not None:
                # Look for follow-through in frames after release
                for i in range(release_frame_idx + 1, len(frames)):
                    frame = frames[i]
                    wrist = frame[16]
                    elbow = frame[14]
                    shoulder = frame[12]
                    
                    if wrist['visibility'] > 0.5 and elbow['visibility'] > 0.5 and shoulder['visibility'] > 0.5:
                        # Check if arm is extended (elbow angle close to 180 degrees)
                        elbow_angle = self._calculate_angle(shoulder, elbow, wrist)
                        if elbow_angle is not None and elbow_angle > 150:
                            return frame
            
            return frames[-1]  # Fallback to last frame if no clear follow-through detected

        else:
            raise ValueError(f"Unknown stage: {stage_name}")

    def _calculate_angle(self, point1, point2, point3):
        """Calculate the angle between three points."""
        if not all(p['visibility'] > 0.5 for p in [point1, point2, point3]):
            return None
        
        # Convert points to numpy arrays
        p1 = np.array([point1['x'], point1['y']])
        p2 = np.array([point2['x'], point2['y']])
        p3 = np.array([point3['x'], point3['y']])
        
        # Calculate vectors
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure value is in valid range
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle

    def _analyze_stage(self, pose_data, stage_name):
        """Extract features for a given stage of the shot and compare to Curry's form.
        
        Args:
            pose_data (dict): The pose data dictionary containing 'frames'.
            stage_name (str): The stage name (e.g., 'stance', 'bend', 'release', 'follow_through').
        
        Returns:
            dict: Analysis results for this stage, including features, similarity score, and recommendations.
        """
        # Find the appropriate frame for this stage
        frame = self._find_stage_frame(pose_data, stage_name)
        if frame is None:
            return None

        # Extract features from the frame
        features = extract_features_from_frame(frame)
        if features is None:
            return None

        # Remove visibility from features and map feature names
        features.pop('visibility', None)
        mapped_features = {self.feature_mapping[k]: v for k, v in features.items()}

        # Convert features to array and scale
        feature_array = np.array([list(mapped_features.values())])
        scaled_features = self.scaler.transform(feature_array)

        # Get anomaly score from Isolation Forest
        anomaly_score = self.model.score_samples(scaled_features)[0]

        # Generate the analysis for this stage
        analysis = self._generate_analysis(mapped_features, None)
        normalized_score = np.exp(anomaly_score + 0.15) * 100
        similarity_score = max(0, min(100, normalized_score))
        analysis['similarity_score'] = float(similarity_score)
        analysis['anomaly_score'] = float(anomaly_score)
        analysis['stage'] = stage_name

        return analysis

    def analyze_shot(self, video_path, user_name, method='ml'):
        """
        Analyze a user's shot and compare it to Curry's form
        
        Args:
            video_path (str): Path to the video file
            user_name (str): Name of the user
            method (str): Analysis method - either 'direct' (feature comparison) or 'ml' (machine learning)
        """
        print(f"\nAnalyzing shot for {user_name}...")
        print("="*50)
        print(f"Using {method.upper()} analysis method")
        
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
        
        # Create a directory to save extracted frames
        frames_dir = os.path.join(self.processed_dir, "frames", user_name)
        frames_dir = os.path.join("data/processed/frames", user_name)
        os.makedirs(frames_dir, exist_ok=True)
        
        # First, find frames for all stages
        print("\nDetecting shot stages...")
        stage_frames = {}
        for stage in ['stance', 'bend', 'release', 'follow_through']:
            print(f"Finding {stage} stage...")
            frame = self._find_stage_frame(pose_data, stage)
            if frame is not None:
                stage_frames[stage] = frame
        
        # Validate stage sequence
        if not self._validate_stage_sequence(stage_frames, pose_data['frames']):
            print("Warning: Shot stages were not detected in the correct order. Analysis may be inaccurate.")
        
        # Analyze each stage
        stage_analyses = {}
        for stage, frame in stage_frames.items():
            print(f"Analyzing {stage} stage...")
            stage_analysis = self._analyze_stage(pose_data, stage)
            if stage_analysis:
                stage_analyses[stage] = stage_analysis
                
                # Draw keypoints on a blank image and save
                pose_img = self.draw_pose_keypoints(frame)
                frame_path = os.path.join(frames_dir, f"{stage}_frame.jpg")
                cv2.imwrite(frame_path, pose_img)
        
        # Calculate overall similarity score (average of all stages)
        overall_similarity = sum(analysis['similarity_score'] for analysis in stage_analyses.values()) / len(stage_analyses)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = self.results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Save JSON results
        json_path = os.path.join(results_dir, f"{user_name}_{timestamp}_analysis.json")
        with open(json_path, 'w') as f:
            json.dump({
                'overall_similarity': overall_similarity,
                'stages': stage_analyses,
                'stage_sequence': {stage: i for i, stage in enumerate(stage_frames.keys())}
            }, f, indent=4)
        
        # Save text recommendations
        txt_path = self._save_recommendations_txt(stage_analyses, user_name, timestamp)
        
        print("\nAnalysis complete!")
        print(f"Results saved to: {json_path}")
        print(f"Recommendations saved to: {txt_path}")
        print("\nSummary:")
        print(f"Analysis method: {method.upper()}")
        print(f"Overall similarity to Curry's form: {overall_similarity:.1f}%")
        for stage, analysis in stage_analyses.items():
            print(f"{stage.capitalize()} stage similarity: {analysis['similarity_score']:.1f}%")
        
        return json_path, txt_path
    
    def _validate_stage_sequence(self, stage_frames, all_frames):
        """Validate that the detected stages are in the correct chronological order.
        
        Args:
            stage_frames (dict): Dictionary mapping stage names to their detected frames
            all_frames (list): List of all frames in the video
        
        Returns:
            bool: True if stages are in correct order, False otherwise
        """
        # Get frame indices for each stage
        stage_indices = {}
        for stage, frame in stage_frames.items():
            # Find the index of this frame in all_frames
            for i, f in enumerate(all_frames):
                if f == frame:  # This is a simple equality check, might need to be more sophisticated
                    stage_indices[stage] = i
                    break
        
        # Check if we have all stages
        if len(stage_indices) != 4:
            print(f"Warning: Only detected {len(stage_indices)} out of 4 stages")
            return False
        
        # Validate order: stance -> bend -> release -> follow_through
        expected_order = ['stance', 'bend', 'release', 'follow_through']
        for i in range(len(expected_order) - 1):
            current_stage = expected_order[i]
            next_stage = expected_order[i + 1]
            if stage_indices[current_stage] >= stage_indices[next_stage]:
                print(f"Warning: {current_stage} stage occurs after {next_stage} stage")
                return False
        
        return True

    def _generate_analysis(self, features, similarity_score):
        """Generate detailed analysis of the shot"""
        # Load Curry's reference statistics
        with open('data/reference/reference_shots/steph_curry_analysis.json', 'r') as f:
            reference_stats = json.load(f)['feature_statistics']
        
        analysis = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'similarity_score': 0.0,  # Will be updated later if None
            'feature_analysis': {},
            'recommendations': []
        }
        
        # Analyze each feature
        for feature, value in features.items():
            ref_stats = reference_stats[feature]
            z_score = abs(value - ref_stats['mean']) / ref_stats['std']
            
            # Calculate feature score (0-100)
            # Use exponential decay based on z-score
            # z-score of 0 = 100, z-score of 2 = ~13.5, z-score of 3 = ~5
            feature_score = 100 * np.exp(-0.5 * z_score)
            
            analysis['feature_analysis'][feature] = {
                'value': float(value),
                'reference_mean': float(ref_stats['mean']),
                'reference_std': float(ref_stats['std']),
                'score': float(feature_score),
                'z_score': float(z_score)
            }
            
            # Generate recommendations for significant differences
            if z_score > 1.5:  # More than 1.5 standard deviations from mean
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
        
        # Update similarity score if provided
        if similarity_score is not None:
            analysis['similarity_score'] = float(similarity_score)
        
        return analysis

    # --- BEGIN PATCH FOR SAVING POSE KEYPOINTS AS IMAGE ---
    # Helper function to draw keypoints on a blank image
    @staticmethod
    def draw_pose_keypoints(frame, image_size=(480, 640), point_radius=5, color=(0, 255, 0)):
        import cv2
        import numpy as np
        blank = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * 255  # white background
        h, w = image_size
        # Assume frame is a list of dicts with 'x', 'y', 'visibility'
        for kp in frame:
            if kp['visibility'] > 0.5:
                x = int(kp['x'] * w)
                y = int(kp['y'] * h)
                cv2.circle(blank, (x, y), point_radius, color, -1)
        return blank
    # --- END PATCH ---

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze a basketball shot against Steph Curry\'s form')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('user_name', help='Name of the user')
    parser.add_argument('--method', choices=['direct', 'ml'], default='ml',
                      help='Analysis method: direct (feature comparison) or ml (machine learning)')
    args = parser.parse_args()
    
    analyzer = UserShotAnalyzer()
    analyzer.analyze_shot(args.video_path, args.user_name, args.method)

if __name__ == "__main__":
    main() 