import os
import json
import librosa
import soundfile as sf
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.target_sr = 16000  # 16kHz for Whisper
        
    def normalize_audio(self, audio_path: Path, target_duration: Optional[float] = None) -> Path:
        """
        Normalize audio file to consistent format.
        
        Args:
            audio_path: Path to input audio file
            target_duration: Optional duration to trim/pad to (in seconds)
            
        Returns:
            Path to normalized audio file
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.target_sr)
            
            # Normalize volume
            audio = librosa.util.normalize(audio)
            
            # Trim silence from beginning and end
            audio, _ = librosa.effects.trim(audio, top_db=20)
            
            # Apply target duration if specified
            if target_duration:
                target_samples = int(target_duration * self.target_sr)
                
                if len(audio) > target_samples:
                    # Trim to target duration
                    audio = audio[:target_samples]
                elif len(audio) < target_samples:
                    # Pad with silence
                    padding = target_samples - len(audio)
                    audio = np.pad(audio, (0, padding), mode='constant')
            
            # Save normalized audio
            output_path = audio_path.parent / f"{audio_path.stem}_normalized.wav"
            sf.write(output_path, audio, self.target_sr)
            
            logger.info(f"Normalized audio: {output_path.name} ({len(audio)/self.target_sr:.1f}s)")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to normalize {audio_path}: {e}")
            raise
    
    def clean_transcription_text(self, text: str, language: str = 'en') -> str:
        """
        Clean transcription text for better quality evaluation.
        
        Args:
            text: Raw transcription text
            language: Language code ('en' or 'hi')
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove timestamps and formatting
        text = re.sub(r'\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[.,]\d{3}', '', text)
        
        # Remove sequence numbers
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove caption formatting
        text = re.sub(r'\[.*?\]', '', text)  # [Music], [Applause], etc.
        text = re.sub(r'\(.*?\)', '', text)  # (inaudible), (laughs), etc.
        
        # Remove speaker labels
        text = re.sub(r'^[A-Z\s]+:\s*', '', text, flags=re.MULTILINE)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Language-specific cleaning
        if language == 'hi':
            # Remove English words mixed in Hindi text (optional)
            # This is aggressive - comment out if mixed language is expected
            # text = re.sub(r'[a-zA-Z]+', '', text)
            
            # Normalize Hindi punctuation
            text = text.replace('।', '.')
            text = text.replace('॥', '.')
        
        # Remove extra punctuation
        text = re.sub(r'[.,!?;:]+', '.', text)
        text = re.sub(r'\.+', '.', text)
        
        # Clean up spacing around punctuation
        text = re.sub(r'\s*\.\s*', '. ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def segment_audio_by_duration(self, audio_path: Path, segment_duration: float = 300) -> List[Path]:
        """
        Segment long audio into smaller chunks.
        
        Args:
            audio_path: Path to input audio
            segment_duration: Duration of each segment in seconds (default: 5 minutes)
            
        Returns:
            List of paths to audio segments
        """
        try:
            audio, sr = librosa.load(audio_path, sr=self.target_sr)
            total_duration = len(audio) / sr
            
            if total_duration <= segment_duration:
                # No need to segment
                return [audio_path]
            
            segments = []
            segment_samples = int(segment_duration * sr)
            
            for i in range(0, len(audio), segment_samples):
                segment_audio = audio[i:i + segment_samples]
                
                # Skip very short segments (less than 30 seconds)
                if len(segment_audio) < 30 * sr:
                    continue
                
                segment_path = audio_path.parent / f"{audio_path.stem}_segment_{i//segment_samples:02d}.wav"
                sf.write(segment_path, segment_audio, sr)
                segments.append(segment_path)
                
                logger.info(f"Created segment: {segment_path.name} ({len(segment_audio)/sr:.1f}s)")
            
            return segments
            
        except Exception as e:
            logger.error(f"Failed to segment {audio_path}: {e}")
            return [audio_path]
    
    def validate_audio_quality(self, audio_path: Path) -> Dict:
        """
        Validate audio quality metrics.
        
        Returns:
            Dictionary with quality metrics
        """
        try:
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Basic metrics
            duration = len(audio) / sr
            rms_energy = np.sqrt(np.mean(audio**2))
            
            # Signal-to-noise ratio estimation
            # Assume first and last 0.5 seconds are noise
            noise_samples = int(0.5 * sr)
            if len(audio) > 2 * noise_samples:
                noise = np.concatenate([audio[:noise_samples], audio[-noise_samples:]])
                signal = audio[noise_samples:-noise_samples]
                
                noise_power = np.mean(noise**2)
                signal_power = np.mean(signal**2)
                
                if noise_power > 0:
                    snr_db = 10 * np.log10(signal_power / noise_power)
                else:
                    snr_db = float('inf')
            else:
                snr_db = 0
            
            # Dynamic range
            dynamic_range = np.max(np.abs(audio)) - np.min(np.abs(audio))
            
            # Zero crossing rate (speech indicator)
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
            
            quality_metrics = {
                'duration': duration,
                'sample_rate': sr,
                'rms_energy': float(rms_energy),
                'snr_db': float(snr_db),
                'dynamic_range': float(dynamic_range),
                'zero_crossing_rate': float(zcr),
                'file_size_mb': audio_path.stat().st_size / (1024 * 1024)
            }
            
            # Quality assessment
            quality_score = 0
            if snr_db > 20:
                quality_score += 3
            elif snr_db > 10:
                quality_score += 2
            elif snr_db > 5:
                quality_score += 1
            
            if 0.05 < zcr < 0.15:  # Typical speech ZCR range
                quality_score += 2
            
            if rms_energy > 0.01:  # Sufficient volume
                quality_score += 1
            
            quality_metrics['quality_score'] = quality_score
            quality_metrics['quality_level'] = (
                'excellent' if quality_score >= 5 else
                'good' if quality_score >= 3 else
                'fair' if quality_score >= 2 else
                'poor'
            )
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Failed to validate {audio_path}: {e}")
            return {'error': str(e)}
    
    def process_dataset(self, target_duration_per_file: float = 600) -> Dict:
        """
        Process entire dataset: normalize audio, clean transcriptions, validate quality.
        
        Args:
            target_duration_per_file: Target duration per audio file in seconds
            
        Returns:
            Processing summary
        """
        summary = {
            'processed_files': [],
            'errors': [],
            'total_duration': 0,
            'quality_distribution': {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
        }
        
        for language in ['english', 'hindi']:
            lang_dir = self.data_dir / language
            audio_dir = lang_dir / 'audio'
            gt_dir = lang_dir / 'ground_truth'
            
            if not audio_dir.exists():
                logger.warning(f"Audio directory not found: {audio_dir}")
                continue
            
            # Process each audio file
            for audio_file in audio_dir.glob('*.wav'):
                try:
                    logger.info(f"Processing {audio_file.name}")
                    
                    # Normalize audio
                    normalized_path = self.normalize_audio(audio_file)
                    
                    # Validate quality
                    quality_metrics = self.validate_audio_quality(normalized_path)
                    
                    # Segment if too long
                    if quality_metrics.get('duration', 0) > target_duration_per_file:
                        segments = self.segment_audio_by_duration(
                            normalized_path, target_duration_per_file
                        )
                    else:
                        segments = [normalized_path]
                    
                    # Process corresponding transcription
                    gt_file = gt_dir / f"{audio_file.stem}.txt"
                    if gt_file.exists():
                        with open(gt_file, 'r', encoding='utf-8') as f:
                            raw_text = f.read()
                        
                        cleaned_text = self.clean_transcription_text(
                            raw_text, 
                            'hi' if language == 'hindi' else 'en'
                        )
                        
                        # Save cleaned transcription
                        cleaned_gt_file = gt_dir / f"{audio_file.stem}_cleaned.txt"
                        with open(cleaned_gt_file, 'w', encoding='utf-8') as f:
                            f.write(cleaned_text)
                    
                    # Update summary
                    file_info = {
                        'original_file': str(audio_file),
                        'normalized_file': str(normalized_path),
                        'segments': [str(s) for s in segments],
                        'language': language,
                        'quality_metrics': quality_metrics
                    }
                    
                    summary['processed_files'].append(file_info)
                    summary['total_duration'] += quality_metrics.get('duration', 0)
                    
                    quality_level = quality_metrics.get('quality_level', 'poor')
                    summary['quality_distribution'][quality_level] += 1
                    
                except Exception as e:
                    error_info = {
                        'file': str(audio_file),
                        'error': str(e)
                    }
                    summary['errors'].append(error_info)
                    logger.error(f"Failed to process {audio_file}: {e}")
        
        # Save processing summary
        summary_file = self.data_dir / 'processing_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Processing complete. Summary saved to {summary_file}")
        logger.info(f"Total duration: {summary['total_duration']/60:.1f} minutes")
        logger.info(f"Quality distribution: {summary['quality_distribution']}")
        
        return summary