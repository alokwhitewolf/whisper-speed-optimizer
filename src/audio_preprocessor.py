import os
import subprocess
import logging
from pathlib import Path
from typing import Union, List, Tuple
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioPreprocessor:
    def __init__(self, output_dir: str = "processed_audio"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def process_audio_with_speed(self, 
                                input_path: Union[str, Path], 
                                speed: float,
                                output_format: str = "mp3") -> Path:
        """
        Process audio file with specified speed using ffmpeg.
        Speed range: 0.5 to 4.0 (practical range: 1.0 to 2.5)
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Audio file not found: {input_path}")
            
        if not 0.5 <= speed <= 4.0:
            raise ValueError(f"Speed must be between 0.5 and 4.0, got {speed}")
        
        output_filename = f"{input_path.stem}_speed_{speed:.2f}x.{output_format}"
        output_path = self.output_dir / output_filename
        
        # FFmpeg command with atempo filter
        # For speeds > 2.0, we need to chain atempo filters
        if speed <= 2.0:
            filter_complex = f"atempo={speed}"
        else:
            # Chain multiple atempo filters for speed > 2.0
            filter_complex = f"atempo=2.0,atempo={speed/2.0}"
        
        cmd = [
            'ffmpeg', '-y', '-i', str(input_path),
            '-filter:a', filter_complex,
            '-ac', '1',  # Mono audio
            '-b:a', '64k',  # Bitrate
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Processed {input_path.name} at {speed}x speed -> {output_path.name}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            raise
    
    def generate_speed_variants(self, 
                               input_path: Union[str, Path],
                               speeds: List[float] = None) -> List[Tuple[float, Path]]:
        """
        Generate multiple speed variants of an audio file.
        Returns list of (speed, output_path) tuples.
        """
        if speeds is None:
            # Default: coarse scan from 1.0 to 2.5
            speeds = [1.0, 1.5, 2.0, 2.5]
        
        results = []
        for speed in tqdm(speeds, desc="Generating speed variants"):
            try:
                output_path = self.process_audio_with_speed(input_path, speed)
                results.append((speed, output_path))
            except Exception as e:
                logger.error(f"Failed to process at {speed}x: {e}")
                
        return results
    
    def generate_fine_speed_range(self, 
                                 input_path: Union[str, Path],
                                 min_speed: float,
                                 max_speed: float,
                                 step: float = 0.05) -> List[Tuple[float, Path]]:
        """
        Generate audio files with fine-grained speed increments.
        Used for binary search optimization.
        """
        speeds = np.arange(min_speed, max_speed + step, step)
        speeds = np.round(speeds, 2)  # Round to 2 decimal places
        
        return self.generate_speed_variants(input_path, speeds.tolist())
    
    def get_audio_duration(self, audio_path: Union[str, Path]) -> float:
        """Get duration of audio file in seconds."""
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(audio_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except Exception as e:
            logger.error(f"Failed to get duration: {e}")
            return 0.0