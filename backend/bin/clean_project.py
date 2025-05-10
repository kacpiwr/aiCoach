#!/usr/bin/env python3
import os
import shutil
from datetime import datetime, timedelta
import glob

def create_directory_structure():
    """Create the main project directories"""
    directories = [
        'data/raw/videos',           # Raw video files
        'data/processed/pose_data',  # Processed pose data
        'data/processed/analysis',   # Analysis results
        'data/models',              # Trained models
        'data/reference',           # Reference data (Curry's shots)
        'docs',                     # Documentation
        'tests',                    # Test files
        'logs'                      # Log files
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def move_files_to_proper_locations():
    """Move files to their proper locations"""
    # Move models
    if os.path.exists('models'):
        for model_file in glob.glob('models/*.joblib'):
            shutil.move(model_file, 'data/models/')
        # Remove any remaining files in models directory
        for file in glob.glob('models/*'):
            if os.path.isfile(file):
                os.remove(file)
        try:
            os.rmdir('models')
        except OSError:
            print("Warning: Could not remove models directory - it may not be empty")
    
    # Move analysis results
    if os.path.exists('analysis_results'):
        # Keep only the most recent analysis files (last 7 days)
        cutoff_date = datetime.now() - timedelta(days=7)
        for file in glob.glob('analysis_results/*'):
            if os.path.isfile(file):
                file_date = datetime.fromtimestamp(os.path.getctime(file))
                if file_date > cutoff_date:
                    if file.endswith('.json'):
                        shutil.move(file, 'data/processed/analysis/')
                    elif file.endswith('.txt'):
                        shutil.move(file, 'data/processed/analysis/')
                    elif file.endswith('.png'):
                        shutil.move(file, 'data/processed/analysis/')
        try:
            os.rmdir('analysis_results')
        except OSError:
            print("Warning: Could not remove analysis_results directory - it may not be empty")
    
    # Move comparison results
    if os.path.exists('comparison_results'):
        for file in glob.glob('comparison_results/*'):
            if os.path.isfile(file):
                if file.endswith('.json'):
                    shutil.move(file, 'data/processed/analysis/')
                elif file.endswith('.png'):
                    shutil.move(file, 'data/processed/analysis/')
        try:
            os.rmdir('comparison_results')
        except OSError:
            print("Warning: Could not remove comparison_results directory - it may not be empty")
    
    # Move shot data
    if os.path.exists('shot_data'):
        # Create subdirectories in reference
        os.makedirs('data/reference/user_shots', exist_ok=True)
        os.makedirs('data/reference/reference_shots', exist_ok=True)
        
        # Move user shots
        if os.path.exists('shot_data/user_shots'):
            for file in glob.glob('shot_data/user_shots/**/*', recursive=True):
                if os.path.isfile(file):
                    rel_path = os.path.relpath(file, 'shot_data/user_shots')
                    target_dir = os.path.join('data/reference/user_shots', os.path.dirname(rel_path))
                    os.makedirs(target_dir, exist_ok=True)
                    shutil.move(file, os.path.join('data/reference/user_shots', rel_path))
            try:
                shutil.rmtree('shot_data/user_shots')
            except OSError:
                print("Warning: Could not remove shot_data/user_shots directory")
        
        # Move reference shots
        if os.path.exists('shot_data/reference_shots'):
            for file in glob.glob('shot_data/reference_shots/**/*', recursive=True):
                if os.path.isfile(file):
                    rel_path = os.path.relpath(file, 'shot_data/reference_shots')
                    target_dir = os.path.join('data/reference/reference_shots', os.path.dirname(rel_path))
                    os.makedirs(target_dir, exist_ok=True)
                    shutil.move(file, os.path.join('data/reference/reference_shots', rel_path))
            try:
                shutil.rmtree('shot_data/reference_shots')
            except OSError:
                print("Warning: Could not remove shot_data/reference_shots directory")
        
        # Try to remove shot_data directory
        try:
            os.rmdir('shot_data')
        except OSError:
            print("Warning: Could not remove shot_data directory - it may not be empty")

def clean_unnecessary_files():
    """Remove unnecessary files"""
    # Remove old analysis findings
    if os.path.exists('analysis_findings'):
        shutil.rmtree('analysis_findings')
    
    # Remove .DS_Store files
    for ds_store in glob.glob('**/.DS_Store', recursive=True):
        try:
            os.remove(ds_store)
        except OSError:
            print(f"Warning: Could not remove {ds_store}")
    
    # Remove __pycache__ directories
    for pycache in glob.glob('**/__pycache__', recursive=True):
        try:
            shutil.rmtree(pycache)
        except OSError:
            print(f"Warning: Could not remove {pycache}")

def update_gitignore():
    """Update .gitignore file"""
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/
.env/
.venv/
myenv/

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific
data/raw/videos/*
data/processed/pose_data/*
data/processed/analysis/*
data/models/*
!data/raw/videos/.gitkeep
!data/processed/pose_data/.gitkeep
!data/processed/analysis/.gitkeep
!data/models/.gitkeep
logs/*
!logs/.gitkeep
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content.strip())

def create_gitkeep_files():
    """Create .gitkeep files in empty directories"""
    directories = [
        'data/raw/videos',
        'data/processed/pose_data',
        'data/processed/analysis',
        'data/models',
        'data/reference',
        'logs'
    ]
    
    for directory in directories:
        gitkeep_path = os.path.join(directory, '.gitkeep')
        if not os.path.exists(gitkeep_path):
            open(gitkeep_path, 'w').close()

def main():
    print("Cleaning and organizing project structure...")
    print("="*50)
    
    # Create directory structure
    print("\nCreating directory structure...")
    create_directory_structure()
    
    # Move files to proper locations
    print("\nMoving files to proper locations...")
    move_files_to_proper_locations()
    
    # Clean unnecessary files
    print("\nCleaning unnecessary files...")
    clean_unnecessary_files()
    
    # Update .gitignore
    print("\nUpdating .gitignore...")
    update_gitignore()
    
    # Create .gitkeep files
    print("\nCreating .gitkeep files...")
    create_gitkeep_files()
    
    print("\nProject cleanup complete!")
    print("="*50)

if __name__ == "__main__":
    main() 