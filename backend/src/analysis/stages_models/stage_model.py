import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib
from datetime import datetime
from train_curry_model import extract_features_from_frame

# Define the stages
STAGES = ["stance_and_balance", "shooting_pocket", "shot_execution", "follow_through"]

def load_stage_data(stage):
    """Load data for a specific stage."""
    stage_dir = f"data/raw/{stage}"
    all_features = []

    # Ensure the directory exists
    if not os.path.exists(stage_dir):
        print(f"Directory for stage '{stage}' not found: {stage_dir}")
        return []

    # Process each frame file in the stage directory
    frame_files = [f for f in os.listdir(stage_dir) if f.endswith('.json')]
    print(f"\nFound {len(frame_files)} frames for stage '{stage}'")

    for frame_file in frame_files:
        frame_path = os.path.join(stage_dir, frame_file)
        try:
            with open(frame_path, 'r') as f:
                frame_data = json.load(f)
            
            # Extract features from the frame
            features = extract_features_from_frame(frame_data)
            if features is not None:
                all_features.append(features)
            else:
                print(f"- Skipping frame {frame_file} due to poor visibility")
        except Exception as e:
            print(f"Error processing frame {frame_file}: {str(e)}")
            continue

    print(f"Successfully processed {len(all_features)} frames for stage '{stage}'")
    return all_features

def train_stage_model(stage, features):
    """Train a model for a specific stage."""
    print(f"\nTraining model for stage: {stage}")
    print("=" * 50)

    # Convert features to DataFrame
    features_df = pd.DataFrame(features)

    # Remove visibility column if present
    if 'visibility' in features_df.columns:
        features_df = features_df.drop('visibility', axis=1)

    # Verify that the correct features are used for training
    expected_features = ['elbow_angle', 'shoulder_angle', 'knee_angle', 'wrist_height', 'elbow_height', 'depth', 'shot_arc', 'release_y']
    if not all(feature in features_df.columns for feature in expected_features):
        print(f"Warning: Some expected features are missing for stage '{stage}'.")
        print("Expected features:", expected_features)
        print("Actual features:", features_df.columns.tolist())
        return None, None

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)

    # Train Isolation Forest for anomaly detection
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,  # Expect 5% of frames to be anomalies
        random_state=42
    )
    model.fit(X_scaled)

    # Validate the model
    predictions = model.predict(X_scaled)
    anomaly_count = sum(predictions == -1)
    print(f"Validation: {anomaly_count} anomalies detected in the training data for stage '{stage}'")

    # Save the model and scaler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"data/models/{stage}"
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"{stage}_model_{timestamp}.joblib")
    scaler_path = os.path.join(model_dir, f"{stage}_scaler_{timestamp}.joblib")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Model and scaler saved for stage '{stage}':")
    print(f"Model: {model_path}")
    print(f"Scaler: {scaler_path}")

    return model, scaler

def train_all_stage_models():
    """Train models for all stages."""
    for stage in STAGES:
        print(f"\nProcessing stage: {stage}")
        features = load_stage_data(stage)
        if len(features) == 0:
            print(f"No valid data found for stage '{stage}'. Skipping...")
            continue
        train_stage_model(stage, features)

if __name__ == "__main__":
    train_all_stage_models()