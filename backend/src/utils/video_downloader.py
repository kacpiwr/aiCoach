import os
import yt_dlp
from datetime import datetime

class VideoDownloader:
    def __init__(self):
        self.download_dir = "nba_videos"
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)

    def download_video(self, url, player_name, quality='720p'):
        """
        Download a video from YouTube
        
        Args:
            url (str): YouTube video URL
            player_name (str): Name of the player (for file naming)
            quality (str): Video quality ('720p', '1080p', etc.)
        
        Returns:
            str: Path to the downloaded video file
        """
        # Configure yt-dlp options
        ydl_opts = {
            'format': f'best[height<={quality[:-1]}]',  # Best quality up to specified height
            'outtmpl': os.path.join(self.download_dir, f'{player_name}_%(id)s.%(ext)s'),
            'quiet': False,
            'no_warnings': False,
            'progress': True,
        }
        
        try:
            print(f"\nDownloading video for {player_name}...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Get video info
                info = ydl.extract_info(url, download=False)
                video_title = info.get('title', 'unknown_title')
                print(f"Video title: {video_title}")
                
                # Download the video
                ydl.download([url])
                
                # Get the downloaded file path
                downloaded_file = os.path.join(
                    self.download_dir,
                    f"{player_name}_{info['id']}.{info['ext']}"
                )
                
                print(f"Successfully downloaded: {downloaded_file}")
                return downloaded_file
                
        except Exception as e:
            print(f"Error downloading video: {e}")
            return None

def main():
    # Example usage
    downloader = VideoDownloader()
    
    # Example videos to download
    videos = [
        {
            'url': 'https://www.youtube.com/watch?v=mzNYxXigRYk&list=PLtwefnFeAxbHjW548-7uwbzCOWJAjt7Dy&index=5',
            'player': 'buddy_hield',
            'quality': '720p'
        },
        {
            'url': 'https://www.youtube.com/watch?v=Mx4H3OWD0KM&list=PLtwefnFeAxbHjW548-7uwbzCOWJAjt7Dy&index=1 ',
            'player': 'anthony_edwards',
            'quality': '720p'
        }
    ]
    
    # Download each video
    for video in videos:
        downloaded_file = downloader.download_video(
            video['url'],
            video['player'],
            video['quality']
        )
        
        if downloaded_file:
            print(f"Video saved to: {downloaded_file}")
        else:
            print(f"Failed to download video for {video['player']}")

if __name__ == "__main__":
    main() 