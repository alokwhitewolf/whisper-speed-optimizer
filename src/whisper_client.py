import os
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Union, Tuple
from openai import OpenAI
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WhisperClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.transcription_cache = {}
        
    def transcribe_audio(self, 
                        audio_path: Union[str, Path],
                        language: Optional[str] = None,
                        model: str = "whisper-1",
                        response_format: str = "json") -> Dict:
        """
        Transcribe audio file using OpenAI Whisper API.
        
        Args:
            audio_path: Path to audio file
            language: ISO-639-1 language code (en, hi)
            model: Whisper model to use
            response_format: Output format (json, text, srt, vtt)
            
        Returns:
            Dictionary with transcription results and metadata
        """
        audio_path = Path(audio_path)
        cache_key = f"{audio_path.name}_{language}_{model}"
        
        # Check cache first
        if cache_key in self.transcription_cache:
            logger.info(f"Using cached transcription for {audio_path.name}")
            return self.transcription_cache[cache_key]
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Get file size for cost estimation
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        
        try:
            start_time = time.time()
            
            with open(audio_path, "rb") as audio_file:
                if language:
                    response = self.client.audio.transcriptions.create(
                        model=model,
                        file=audio_file,
                        language=language,
                        response_format=response_format
                    )
                else:
                    response = self.client.audio.transcriptions.create(
                        model=model,
                        file=audio_file,
                        response_format=response_format
                    )
            
            transcription_time = time.time() - start_time
            
            # Parse response based on format
            if response_format == "json":
                transcription_text = response.text if hasattr(response, 'text') else str(response)
            else:
                transcription_text = str(response)
            
            result = {
                "transcription": transcription_text,
                "audio_file": audio_path.name,
                "language": language,
                "model": model,
                "transcription_time": transcription_time,
                "file_size_mb": file_size_mb,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache the result
            self.transcription_cache[cache_key] = result
            
            # Save transcription to file for analysis
            transcription_log_dir = Path("transcription_logs")
            transcription_log_dir.mkdir(exist_ok=True)
            log_file = transcription_log_dir / f"{audio_path.stem}.txt"
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"File: {audio_path.name}\n")
                f.write(f"Language: {language}\n")
                f.write(f"Model: {model}\n")
                f.write(f"Duration: {transcription_time:.2f}s\n")
                f.write(f"File size: {file_size_mb:.2f}MB\n")
                f.write("="*50 + "\n")
                f.write(transcription_text)
                f.write("\n")
            
            logger.info(f"Transcribed {audio_path.name} in {transcription_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {e}")
            raise
    
    def batch_transcribe(self, 
                        audio_paths: list,
                        language: Optional[str] = None) -> list:
        """
        Transcribe multiple audio files.
        
        Returns:
            List of transcription results
        """
        results = []
        
        for audio_path in audio_paths:
            try:
                result = self.transcribe_audio(audio_path, language)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to transcribe {audio_path}: {e}")
                results.append({
                    "audio_file": Path(audio_path).name,
                    "error": str(e),
                    "transcription": None
                })
        
        return results
    
    def estimate_cost(self, audio_duration_minutes: float) -> float:
        """
        Estimate transcription cost based on audio duration.
        Whisper pricing: $0.006 per minute
        """
        cost_per_minute = 0.006
        return audio_duration_minutes * cost_per_minute
    
    def save_transcriptions(self, results: list, output_path: Union[str, Path]):
        """Save transcription results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved transcriptions to {output_path}")