import os
import shutil
from datetime import datetime

def rename_videos():
    print("\nRenaming videos to consistent format...")
    print("="*50)
    
    videos_dir = "data/raw/nba_videos/cut_shots"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create backup directory
    backup_dir = os.path.join(videos_dir, "backup_" + timestamp)
    os.makedirs(backup_dir, exist_ok=True)
    
    # Get all video files
    video_files = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]
    moj_film_count = sum(1 for f in video_files if f.startswith('Mój Film'))
    
    print(f"\nFound {moj_film_count} videos to rename")
    
    # Rename files
    renamed_count = 0
    for i, video_file in enumerate(video_files):
        if video_file.startswith('Mój Film'):
            # Create new name
            new_name = f"steph_curry_additional_shot_{i+1:03d}.mp4"
            old_path = os.path.join(videos_dir, video_file)
            new_path = os.path.join(videos_dir, new_name)
            backup_path = os.path.join(backup_dir, video_file)
            
            try:
                # Backup original file
                shutil.copy2(old_path, backup_path)
                # Rename file
                os.rename(old_path, new_path)
                print(f"Renamed: {video_file} -> {new_name}")
                renamed_count += 1
            except Exception as e:
                print(f"Error renaming {video_file}: {str(e)}")
    
    print(f"\nRenaming complete!")
    print(f"Successfully renamed {renamed_count} out of {moj_film_count} videos")
    print(f"Original files backed up to: {backup_dir}")
    print("="*50)

if __name__ == "__main__":
    rename_videos() 