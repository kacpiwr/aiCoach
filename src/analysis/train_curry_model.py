import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib
import pandas as pd
from datetime import datetime
import cv2
from .pose_analyzer import PoseAnalyzer

def calculate_angle(point1, point2, point3):
    """Calculate angle between three points"""
    a = np.array([point1['x'], point1['y']])
    b = np.array([point2['x'], point2['y']])
    c = np.array([point3['x'], point3['y']])
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def extract_features_from_frame(frame):
    """Extract shooting form features from a single frame of pose data"""
    # MediaPipe indices for key points
    RIGHT_SHOULDER = 12
    RIGHT_ELBOW = 14
    RIGHT_WRIST = 16
    RIGHT_HIP = 24
    RIGHT_KNEE = 26
    RIGHT_ANKLE = 28
    
    # Extract key points
    shoulder = frame[RIGHT_SHOULDER]
    elbow = frame[RIGHT_ELBOW]
    wrist = frame[RIGHT_WRIST]
    hip = frame[RIGHT_HIP]
    knee = frame[RIGHT_KNEE]
    ankle = frame[RIGHT_ANKLE]
    
    # Check visibility of key points for shot analysis
    key_points_visibility = [
        shoulder['visibility'],
        elbow['visibility'],
        wrist['visibility']  # Most important for shot analysis
    ]
    
    # Calculate average visibility of key points
    visibility = np.mean(key_points_visibility)
    
    # If key points aren't visible enough, return None
    if visibility < 0.5:  # More lenient threshold, focusing on upper body
        return None
    
    # Calculate angles
    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    shoulder_angle = calculate_angle(hip, shoulder, elbow)
    knee_angle = calculate_angle(hip, knee, ankle)
    
    # Calculate positions relative to shoulder
    wrist_height = wrist['y'] - shoulder['y']
    elbow_height = elbow['y'] - shoulder['y']
    depth = wrist['z'] - shoulder['z']
    
    # Calculate shot arc
    shot_arc = np.arctan2(wrist['y'] - shoulder['y'], wrist['x'] - shoulder['x'])
    
    # Calculate release point relative to shoulder
    release_y = wrist['y'] - shoulder['y']
    
    return {
        'elbow_angle': elbow_angle,
        'shoulder_angle': shoulder_angle,
        'knee_angle': knee_angle,
        'wrist_height': wrist_height,
        'elbow_height': elbow_height,
        'depth': depth,
        'shot_arc': shot_arc,
        'release_y': release_y,
        'visibility': visibility
    }

def load_shot_data():
    """Load all shot data from cut shot videos"""
    videos_dir = "data/raw/nba_videos/cut_shots"
    all_features = []
    analyzer = PoseAnalyzer()
    
    # Get all Steph Curry video files
    video_files = [f for f in os.listdir(videos_dir) if f.startswith('steph_curry_') and f.endswith('.mp4')]
    total_videos = len(video_files)
    print(f"\nFound {total_videos} Steph Curry videos to process")
    
    # Process each video
    for i, video_file in enumerate(video_files, 1):
        print(f"\nProcessing video {i}/{total_videos}: {video_file}...")
        video_path = os.path.join(videos_dir, video_file)
        
        try:
            # Analyze the video to get pose data
            analysis_path = analyzer.analyze_shot(
                video_path,
                is_reference=True,
                player_name="steph_curry"
            )
            
            # Load the pose data
            with open(analysis_path, 'r') as f:
                data = json.load(f)
            
            # Get the frame with highest wrist position (release point)
            best_frame = None
            highest_wrist = -float('inf')
            
            for frame in data['frames']:
                try:
                    wrist = frame[16]  # Index 16 is right wrist
                    if isinstance(wrist, dict) and 'y' in wrist and 'visibility' in wrist:
                        wrist_y = wrist['y']
                        if wrist_y > highest_wrist and wrist['visibility'] > 0.5:
                            highest_wrist = wrist_y
                            best_frame = frame
                except (IndexError, KeyError, TypeError) as e:
                    continue
            
            if best_frame is not None:
                print(f"- Found best frame with wrist height: {highest_wrist:.3f}")
                features = extract_features_from_frame(best_frame)
                if features is not None:
                    print("- Frame has good visibility, adding to dataset")
                    all_features.append(features)
                    print(f"- Progress: {len(all_features)}/{total_videos} shots processed")
                else:
                    print("- Frame has poor visibility of key points")
            else:
                print("- No valid frame found with required keypoints")
                
        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")
            continue
    
    print(f"\nSuccessfully processed {len(all_features)} out of {total_videos} shots")
    return all_features

def train_model():
    print("\nTraining Curry Shot Analysis Model...")
    print("="*50)
    
    # Load and prepare data
    print("Loading shot data...")
    shot_data = load_shot_data()
    print(f"Loaded data from {len(shot_data)} shots")
    
    if len(shot_data) == 0:
        print("No valid shots found in the data!")
        return
    
    # Convert to DataFrame
    features_df = pd.DataFrame(shot_data)
    
    # Remove visibility column from training data
    X = features_df.drop('visibility', axis=1)
    
    # Verify that the correct features are used for training
    expected_features = ['elbow_angle', 'shoulder_angle', 'knee_angle', 'wrist_height', 'elbow_height', 'depth', 'shot_arc', 'release_y']
    if not all(feature in X.columns for feature in expected_features):
        print("Warning: Some expected features are missing from the training data.")
        print("Expected features:", expected_features)
        print("Actual features:", X.columns.tolist())
    else:
        print("All expected features are present in the training data.")
    
    # Calculate feature statistics for reference
    feature_stats = {
        'mean': X.mean(),
        'std': X.std(),
        'min': X.min(),
        'max': X.max()
    }
    
    # Save feature statistics
    stats_dir = "data/reference/reference_shots"
    os.makedirs(stats_dir, exist_ok=True)
    with open(os.path.join(stats_dir, 'steph_curry_analysis.json'), 'w') as f:
        json.dump({
            'feature_statistics': {
                col: {
                    'mean': float(feature_stats['mean'][col]),
                    'std': float(feature_stats['std'][col]),
                    'min': float(feature_stats['min'][col]),
                    'max': float(feature_stats['max'][col])
                }
                for col in X.columns
            }
        }, f, indent=4)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest for anomaly detection
    print("Training Isolation Forest model...")
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,  # Expect 5% of shots to be significantly different
        random_state=42
    )
    model.fit(X_scaled)
    
    # Simple validation: Check if the model can predict anomalies on the training data
    predictions = model.predict(X_scaled)
    anomaly_count = sum(predictions == -1)
    print(f"Model validation: {anomaly_count} anomalies detected in the training data.")
    
    # Save model and scaler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = "data/models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Delete old model and scaler files
    for file in os.listdir(model_dir):
        if file.startswith('curry_shot_model_') or file.startswith('curry_shot_scaler_'):
            os.remove(os.path.join(model_dir, file))
    
    model_path = os.path.join(model_dir, f"curry_shot_model_{timestamp}.joblib")
    scaler_path = os.path.join(model_dir, f"curry_shot_scaler_{timestamp}.joblib")
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\nModel and scaler saved:")
    print(f"Model: {model_path}")
    print(f"Scaler: {scaler_path}")
    
    return model, scaler, feature_stats

if __name__ == "__main__":
    train_model() 