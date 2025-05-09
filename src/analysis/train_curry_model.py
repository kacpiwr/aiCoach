import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
from datetime import datetime

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
    horizontal_distance = wrist['x'] - shoulder['x']
    depth = wrist['z'] - shoulder['z']
    
    # Calculate shot arc
    shot_arc = np.arctan2(wrist['y'] - shoulder['y'], wrist['x'] - shoulder['x'])
    
    # Calculate release point relative to shoulder
    release_x = wrist['x'] - shoulder['x']
    release_y = wrist['y'] - shoulder['y']
    release_z = wrist['z'] - shoulder['z']
    
    return {
        'elbow_angle': elbow_angle,
        'shoulder_angle': shoulder_angle,
        'knee_angle': knee_angle,
        'wrist_height': wrist_height,
        'elbow_height': elbow_height,
        'horizontal_distance': horizontal_distance,
        'depth': depth,
        'shot_arc': shot_arc,
        'release_x': release_x,
        'release_y': release_y,
        'release_z': release_z,
        'visibility': visibility
    }

def load_shot_data():
    """Load all shot data from JSON files"""
    data_dir = "shot_data/reference_shots"
    all_features = []
    
    # Load all JSON files
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            print(f"\nProcessing {filename}...")
            with open(os.path.join(data_dir, filename), 'r') as f:
                try:
                    data = json.load(f)
                    print(f"- Found {len(data['frames'])} frame lists")
                    
                    # Get the frame with highest wrist position (release point) across all frames
                    best_frame = None
                    highest_wrist = -float('inf')
                    
                    # Look through all frame lists
                    for i, frame_list in enumerate(data['frames']):
                        print(f"- Processing frame list {i+1}/{len(data['frames'])}")
                        print(f"  - Frame list length: {len(frame_list)}")
                        
                        # Debug: print first frame structure
                        if i == 0:
                            print("  - First frame structure:")
                            print(frame_list)
                        
                        # Each frame_list is a list of keypoints
                        if len(frame_list) >= 33:  # Each frame should have 33 keypoints
                            # The frame_list itself is a single frame with keypoints
                            try:
                                wrist = frame_list[16]  # Index 16 is right wrist
                                if isinstance(wrist, dict) and 'y' in wrist and 'visibility' in wrist:
                                    wrist_y = wrist['y']
                                    if wrist_y > highest_wrist and wrist['visibility'] > 0.5:  # Check wrist visibility
                                        highest_wrist = wrist_y
                                        best_frame = frame_list
                                        print(f"  - Found new best frame with wrist height: {wrist_y:.3f}")
                            except (IndexError, KeyError, TypeError) as e:
                                print(f"  - Error processing frame: {str(e)}")
                                continue
                        else:
                            print(f"  - Frame list too short: {len(frame_list)} keypoints")
                    
                    if best_frame is not None:
                        print("- Found best frame with wrist height:", highest_wrist)
                        features = extract_features_from_frame(best_frame)
                        if features is not None:  # Only add if features were successfully extracted
                            print("- Frame has good visibility, adding to dataset")
                            all_features.append(features)
                        else:
                            print("- Frame has poor visibility of key points")
                    else:
                        print("- No valid frame found with required keypoints")
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    continue
    
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
    
    # Since this is reference data, we'll use a simple binary classification
    # 1 for all shots (they're all Curry's shots)
    y = np.ones(len(X))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    print(f"\nModel Performance:")
    print(f"Training accuracy: {train_score:.3f}")
    print(f"Testing accuracy: {test_score:.3f}")
    
    # Save model and scaler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, f"curry_shot_model_{timestamp}.joblib")
    scaler_path = os.path.join(model_dir, f"curry_shot_scaler_{timestamp}.joblib")
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    
    # Save feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    importance_path = os.path.join(model_dir, f"feature_importance_{timestamp}.csv")
    feature_importance.to_csv(importance_path, index=False)
    print(f"Feature importance saved to: {importance_path}")
    
    # Print feature importance
    print("\nFeature Importance:")
    for _, row in feature_importance.iterrows():
        print(f"{row['feature']}: {row['importance']:.3f}")
    
    print("\nTraining complete!")
    print("="*50)

if __name__ == "__main__":
    train_model() 