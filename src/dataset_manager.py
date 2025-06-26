#!/usr/bin/env python3
"""
Dataset management module for video collection and audio processing.
"""

import json
import subprocess
from pathlib import Path
from typing import List, Dict

class DatasetManager:
    """Manages video dataset collection and audio processing."""
    
    def __init__(self, config_file: str = "curated_videos.json"):
        """Initialize dataset manager."""
        self.config_file = Path(config_file)
        self.data_dir = Path("data/english")
        self.audio_dir = self.data_dir / "audio"
        self.ground_truth_dir = self.data_dir / "ground_truth"
        self.metadata_dir = self.data_dir / "metadata"
        
        # Create directories
        for dir_path in [self.audio_dir, self.ground_truth_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def collect_videos(self) -> None:
        """Collect videos from curated list."""
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file}")
        
        with open(self.config_file, 'r') as f:
            videos = json.load(f)
        
        print(f"📥 Collecting {len(videos)} videos...")
        
        for video_id, video_info in videos.items():
            print(f"  Processing: {video_info.get('title', video_id)}")
            self._download_video(video_id, video_info)
    
    def _download_video(self, video_id: str, video_info: Dict) -> None:
        """Download individual video and extract audio."""
        try:
            # Download audio
            audio_output = self.audio_dir / f"{video_id}.wav"
            if not audio_output.exists():
                cmd = [
                    'yt-dlp',
                    '--extract-audio',
                    '--audio-format', 'wav',
                    '--audio-quality', '0',
                    '-o', str(self.audio_dir / f"{video_id}.%(ext)s"),
                    f"https://youtube.com/watch?v={video_id}"
                ]
                subprocess.run(cmd, check=True, capture_output=True)
            
            # Download subtitles
            subtitle_output = self.ground_truth_dir / f"{video_id}.en.txt"
            if not subtitle_output.exists():
                cmd = [
                    'yt-dlp',
                    '--write-subs',
                    '--write-auto-subs',
                    '--sub-langs', 'en',
                    '--skip-download',
                    '-o', str(self.ground_truth_dir / f"{video_id}.%(ext)s"),
                    f"https://youtube.com/watch?v={video_id}"
                ]
                subprocess.run(cmd, check=True, capture_output=True)
            
            # Normalize audio
            normalized_output = self.audio_dir / f"{video_id}_normalized.wav"
            if not normalized_output.exists():
                self._normalize_audio(audio_output, normalized_output)
                
        except subprocess.CalledProcessError as e:
            print(f"    ❌ Failed to download {video_id}: {e}")
        except Exception as e:
            print(f"    ❌ Error processing {video_id}: {e}")
    
    def _normalize_audio(self, input_file: Path, output_file: Path) -> None:
        """Normalize audio levels."""
        cmd = [
            'ffmpeg',
            '-i', str(input_file),
            '-af', 'loudnorm',
            '-y', str(output_file)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    
    def process_audio(self) -> None:
        """Process audio files for speed experiments."""
        print("🎵 Processing audio files...")
        
        # Get list of normalized audio files
        audio_files = list(self.audio_dir.glob("*_normalized.wav"))
        
        if not audio_files:
            print("❌ No normalized audio files found. Run collect_videos() first.")
            return
        
        print(f"Found {len(audio_files)} audio files to process")
        
        # For now, just report what would be processed
        # The actual speed processing will be done during experiments
        for audio_file in audio_files:
            print(f"  Ready for processing: {audio_file.name}")
    
    def get_video_list(self) -> List[str]:
        """Get list of available videos."""
        audio_files = list(self.audio_dir.glob("*_normalized.wav"))
        return [f.stem.replace("_normalized", "") for f in audio_files]
    
    def validate_dataset(self) -> Dict:
        """Validate dataset completeness."""
        video_list = self.get_video_list()
        
        validation_results = {
            'total_videos': len(video_list),
            'missing_audio': [],
            'missing_subtitles': [],
            'ready_for_analysis': []
        }
        
        for video_id in video_list:
            # Check audio
            audio_file = self.audio_dir / f"{video_id}_normalized.wav"
            if not audio_file.exists():
                validation_results['missing_audio'].append(video_id)
                continue
            
            # Check subtitles
            subtitle_file = self.ground_truth_dir / f"{video_id}.en.txt"
            if not subtitle_file.exists():
                validation_results['missing_subtitles'].append(video_id)
                continue
            
            validation_results['ready_for_analysis'].append(video_id)
        
        return validation_results