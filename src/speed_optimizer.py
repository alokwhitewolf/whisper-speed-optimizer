import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Callable
import json
import time
from datetime import datetime

from .audio_preprocessor import AudioPreprocessor
from .whisper_client import WhisperClient
from .quality_evaluator import QualityEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeedOptimizer:
    def __init__(self, 
                 whisper_client: WhisperClient,
                 audio_preprocessor: AudioPreprocessor,
                 quality_evaluator: QualityEvaluator):
        self.whisper = whisper_client
        self.preprocessor = audio_preprocessor
        self.evaluator = quality_evaluator
        self.search_history = []
        
    def binary_search_optimal_speed(self,
                                  audio_path: Path,
                                  ground_truth: str,
                                  language: str = 'en',
                                  min_speed: float = 1.0,
                                  max_speed: float = 2.5,
                                  quality_threshold: float = 5.0,
                                  precision: float = 0.1) -> Dict:
        """
        Find optimal speed using binary search.
        
        Args:
            audio_path: Path to original audio file
            ground_truth: Reference transcription
            language: Language code (en, hi)
            min_speed: Minimum speed to test
            max_speed: Maximum speed to test
            quality_threshold: Maximum acceptable WER increase (%)
            precision: Speed precision (default 0.05x)
            
        Returns:
            Dict with optimal speed and quality metrics
        """
        logger.info(f"Starting binary search for {audio_path.name}")
        logger.info(f"Speed range: [{min_speed}, {max_speed}], Quality threshold: {quality_threshold}%")
        
        # Get baseline quality at 1.0x speed
        baseline_audio = self.preprocessor.process_audio_with_speed(audio_path, 1.0)
        baseline_result = self.whisper.transcribe_audio(baseline_audio, language)
        baseline_metrics = self.evaluator.evaluate_transcription(
            ground_truth, baseline_result['transcription'], language
        )
        
        logger.info(f"Baseline WER: {baseline_metrics['wer_percent']:.2f}%")
        
        # Binary search
        search_log = []
        optimal_speed = min_speed
        optimal_metrics = baseline_metrics
        
        # Limit binary search to max 3 iterations for practical purposes
        iteration_count = 0
        max_iterations = 3
        
        while max_speed - min_speed > precision and iteration_count < max_iterations:
            iteration_count += 1
            mid_speed = round((min_speed + max_speed) / 2, 1)  # Round to 0.1x precision
            
            # Test mid speed
            test_audio = self.preprocessor.process_audio_with_speed(audio_path, mid_speed)
            test_result = self.whisper.transcribe_audio(test_audio, language)
            test_metrics = self.evaluator.evaluate_transcription(
                ground_truth, test_result['transcription'], language
            )
            
            # Calculate degradation
            degradation = self.evaluator.compare_quality_degradation(baseline_metrics, test_metrics)
            wer_increase = degradation.get('wer_increase', float('inf'))
            
            # Log this iteration
            iteration_data = {
                'speed': mid_speed,
                'wer': test_metrics['wer_percent'],
                'wer_increase': wer_increase,
                'cer': test_metrics['cer_percent'],
                'acceptable': wer_increase <= quality_threshold
            }
            search_log.append(iteration_data)
            
            logger.info(f"Speed {mid_speed}x: WER={test_metrics['wer_percent']:.2f}%, "
                       f"Increase={wer_increase:.2f}%")
            
            if wer_increase <= quality_threshold:
                # Quality is acceptable, try faster
                optimal_speed = mid_speed
                optimal_metrics = test_metrics
                min_speed = mid_speed
            else:
                # Quality degraded too much, try slower
                max_speed = mid_speed
        
        # Final result
        result = {
            'audio_file': audio_path.name,
            'language': language,
            'optimal_speed': optimal_speed,
            'baseline_metrics': baseline_metrics,
            'optimal_metrics': optimal_metrics,
            'quality_threshold': quality_threshold,
            'search_iterations': len(search_log),
            'search_log': search_log,
            'cost_reduction': (1 - (1 / optimal_speed)) * 100,
            'time_savings': (1 - (1 / optimal_speed)) * 100,
            'timestamp': datetime.now().isoformat()
        }
        
        self.search_history.append(result)
        return result
    
    def coarse_scan(self,
                   audio_path: Path,
                   ground_truth: str,
                   language: str = 'en',
                   speeds: list = None) -> Dict:
        """
        Perform initial coarse scan to identify promising speed ranges.
        """
        if speeds is None:
            # Reduced coarse scan points for efficiency
            speeds = [1.0, 1.5, 2.0, 2.5]
        
        logger.info(f"Performing coarse scan for {audio_path.name}")
        
        # Get baseline
        baseline_audio = self.preprocessor.process_audio_with_speed(audio_path, 1.0)
        baseline_result = self.whisper.transcribe_audio(baseline_audio, language)
        baseline_metrics = self.evaluator.evaluate_transcription(
            ground_truth, baseline_result['transcription'], language
        )
        
        scan_results = []
        
        for speed in speeds:
            if speed == 1.0:
                scan_results.append({
                    'speed': 1.0,
                    'wer': baseline_metrics['wer_percent'],
                    'cer': baseline_metrics['cer_percent'],
                    'wer_increase': 0.0,
                    'transcription': baseline_result['transcription']
                })
                continue
            
            # Test each speed
            test_audio = self.preprocessor.process_audio_with_speed(audio_path, speed)
            test_result = self.whisper.transcribe_audio(test_audio, language)
            test_metrics = self.evaluator.evaluate_transcription(
                ground_truth, test_result['transcription'], language
            )
            
            degradation = self.evaluator.compare_quality_degradation(baseline_metrics, test_metrics)
            
            scan_results.append({
                'speed': speed,
                'wer': test_metrics['wer_percent'],
                'cer': test_metrics['cer_percent'],
                'wer_increase': degradation.get('wer_increase', float('inf')),
                'transcription': test_result['transcription']
            })
            
            logger.info(f"Speed {speed}x: WER={test_metrics['wer_percent']:.2f}%, "
                       f"Increase={degradation.get('wer_increase', 0):.2f}%")
        
        return {
            'audio_file': audio_path.name,
            'language': language,
            'baseline_metrics': baseline_metrics,
            'scan_results': scan_results,
            'timestamp': datetime.now().isoformat()
        }
    
    def find_quality_cliff(self,
                         scan_results: list,
                         threshold_increase: float = 5.0) -> Tuple[float, float]:
        """
        Analyze coarse scan results to find the quality cliff.
        Returns (min_speed, max_speed) range for binary search.
        """
        # Find the last speed where quality is acceptable
        last_good_speed = 1.0
        first_bad_speed = None
        
        for i, result in enumerate(scan_results):
            if result['wer_increase'] <= threshold_increase:
                last_good_speed = result['speed']
            else:
                first_bad_speed = result['speed']
                break
        
        if first_bad_speed is None:
            # All speeds were acceptable
            return last_good_speed, scan_results[-1]['speed']
        
        # Return range for binary search
        return last_good_speed, first_bad_speed
    
    def optimize_dataset(self,
                        audio_paths: list,
                        ground_truths: Dict[str, str],
                        languages: Dict[str, str],
                        quality_threshold: float = 5.0) -> list:
        """
        Optimize speed for multiple audio files.
        """
        results = []
        
        for audio_path in audio_paths:
            audio_name = Path(audio_path).stem
            
            if audio_name not in ground_truths:
                logger.warning(f"No ground truth found for {audio_name}")
                continue
            
            language = languages.get(audio_name, 'en')
            
            # First, do coarse scan
            coarse_result = self.coarse_scan(
                Path(audio_path),
                ground_truths[audio_name],
                language
            )
            
            # Find promising range
            min_speed, max_speed = self.find_quality_cliff(
                coarse_result['scan_results'],
                quality_threshold
            )
            
            # Binary search in the promising range
            optimal_result = self.binary_search_optimal_speed(
                Path(audio_path),
                ground_truths[audio_name],
                language,
                min_speed,
                max_speed,
                quality_threshold
            )
            
            results.append({
                'coarse_scan': coarse_result,
                'optimization': optimal_result
            })
        
        return results
    
    def save_results(self, results: list, output_path: Path):
        """Save optimization results to JSON."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {output_path}")