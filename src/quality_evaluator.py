import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
from jiwer import wer, cer, mer
import re
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualityEvaluator:
    def __init__(self):
        self.normalization_rules = {
            'lowercase': True,
            'remove_punctuation': True,
            'normalize_numbers': True,
            'remove_extra_spaces': True
        }
    
    def normalize_text(self, text: str, language: str = 'en') -> str:
        """
        Normalize text for fair comparison.
        Handles language-specific normalization and strips metadata headers.
        """
        if not text:
            return ""
        
        # Strip metadata headers from transcriptions
        text = self._strip_metadata_header(text)
        
        # Convert to lowercase
        if self.normalization_rules['lowercase']:
            text = text.lower()
        
        # Remove punctuation
        if self.normalization_rules['remove_punctuation']:
            if language == 'hi':
                # Hindi-specific punctuation
                text = re.sub(r'[।॥,\.!?\-\:\;\"\'\(\)\[\]\{\}]', ' ', text)
            else:
                # English punctuation
                text = re.sub(r'[^\w\s]', ' ', text)
        
        # Normalize numbers
        if self.normalization_rules['normalize_numbers']:
            # Convert digits to words (simplified)
            text = re.sub(r'\d+', ' number ', text)
        
        # Remove extra spaces
        if self.normalization_rules['remove_extra_spaces']:
            text = ' '.join(text.split())
        
        return text.strip()
    
    def _strip_metadata_header(self, text: str) -> str:
        """
        Strip metadata headers from transcription files that can contaminate WER/CER scores.
        
        Example headers:
        File: video_normalized_speed_1.50x.mp3
        Language: en
        Model: whisper-1
        Duration: 9.31s
        File size: 1.09MB
        ==================================================
        """
        # Remove lines starting with common metadata fields
        lines = text.split('\n')
        content_lines = []
        
        skip_patterns = [
            r'^File:\s*',
            r'^Language:\s*',
            r'^Model:\s*',
            r'^Duration:\s*',
            r'^File size:\s*',
            r'^={3,}',  # Lines with 3+ equals signs
            r'^\s*$'    # Empty lines at start
        ]
        
        content_started = False
        for line in lines:
            # Check if this line matches any skip pattern
            should_skip = any(re.match(pattern, line) for pattern in skip_patterns)
            
            if not should_skip and line.strip():
                content_started = True
            
            if content_started and not should_skip:
                content_lines.append(line)
        
        return '\n'.join(content_lines).strip()
    
    def calculate_wer(self, reference: str, hypothesis: str, language: str = 'en') -> float:
        """
        Calculate Word Error Rate (WER).
        WER = (Substitutions + Deletions + Insertions) / Total Words in Reference
        """
        ref_normalized = self.normalize_text(reference, language)
        hyp_normalized = self.normalize_text(hypothesis, language)
        
        if not ref_normalized:
            return 0.0 if not hyp_normalized else 1.0
        
        error_rate = wer(ref_normalized, hyp_normalized)
        return min(error_rate, 1.0)  # Cap at 100%
    
    def calculate_cer(self, reference: str, hypothesis: str, language: str = 'en') -> float:
        """
        Calculate Character Error Rate (CER).
        More granular than WER, especially useful for agglutinative languages.
        """
        ref_normalized = self.normalize_text(reference, language)
        hyp_normalized = self.normalize_text(hypothesis, language)
        
        if not ref_normalized:
            return 0.0 if not hyp_normalized else 1.0
        
        error_rate = cer(ref_normalized, hyp_normalized)
        return min(error_rate, 1.0)
    
    def calculate_mer(self, reference: str, hypothesis: str, language: str = 'en') -> float:
        """
        Calculate Match Error Rate (MER).
        Independent of word order.
        """
        ref_normalized = self.normalize_text(reference, language)
        hyp_normalized = self.normalize_text(hypothesis, language)
        
        if not ref_normalized:
            return 0.0 if not hyp_normalized else 1.0
        
        error_rate = mer(ref_normalized, hyp_normalized)
        return min(error_rate, 1.0)
    
    def calculate_semantic_similarity(self, reference: str, hypothesis: str) -> float:
        """
        Calculate semantic similarity using sequence matching.
        Returns a score between 0 and 1 (1 = identical).
        """
        ref_normalized = self.normalize_text(reference)
        hyp_normalized = self.normalize_text(hypothesis)
        
        matcher = SequenceMatcher(None, ref_normalized, hyp_normalized)
        return matcher.ratio()
    
    def evaluate_transcription(self, 
                             reference: str, 
                             hypothesis: str,
                             language: str = 'en') -> Dict[str, float]:
        """
        Comprehensive evaluation of transcription quality.
        """
        results = {
            'wer': self.calculate_wer(reference, hypothesis, language),
            'cer': self.calculate_cer(reference, hypothesis, language),
            'mer': self.calculate_mer(reference, hypothesis, language),
            'semantic_similarity': self.calculate_semantic_similarity(reference, hypothesis),
            'word_count_ref': len(self.normalize_text(reference, language).split()),
            'word_count_hyp': len(self.normalize_text(hypothesis, language).split()),
            'char_count_ref': len(self.normalize_text(reference, language)),
            'char_count_hyp': len(self.normalize_text(hypothesis, language))
        }
        
        # Add percentage versions
        results['wer_percent'] = results['wer'] * 100
        results['cer_percent'] = results['cer'] * 100
        results['mer_percent'] = results['mer'] * 100
        
        return results
    
    def compare_quality_degradation(self,
                                  baseline_metrics: Dict[str, float],
                                  speed_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate quality degradation between baseline and speed-modified transcription.
        """
        degradation = {}
        
        for metric in ['wer', 'cer', 'mer']:
            if metric in baseline_metrics and metric in speed_metrics:
                # Calculate relative increase in error rate
                baseline_val = baseline_metrics[metric]
                speed_val = speed_metrics[metric]
                
                if baseline_val == 0:
                    # If baseline is perfect, any error is infinite degradation
                    degradation[f'{metric}_increase'] = float('inf') if speed_val > 0 else 0
                else:
                    # Percentage increase in error
                    degradation[f'{metric}_increase'] = ((speed_val - baseline_val) / baseline_val) * 100
                
                # Absolute increase
                degradation[f'{metric}_absolute_increase'] = speed_val - baseline_val
        
        # Semantic similarity decrease
        if 'semantic_similarity' in baseline_metrics and 'semantic_similarity' in speed_metrics:
            degradation['semantic_similarity_decrease'] = (
                baseline_metrics['semantic_similarity'] - speed_metrics['semantic_similarity']
            ) * 100
        
        return degradation
    
    def is_quality_acceptable(self,
                            degradation: Dict[str, float],
                            wer_threshold: float = 5.0,
                            cer_threshold: float = 5.0) -> bool:
        """
        Determine if quality degradation is within acceptable limits.
        
        Args:
            degradation: Quality degradation metrics
            wer_threshold: Maximum acceptable WER increase (percentage)
            cer_threshold: Maximum acceptable CER increase (percentage)
        """
        wer_increase = degradation.get('wer_increase', float('inf'))
        cer_increase = degradation.get('cer_increase', float('inf'))
        
        return wer_increase <= wer_threshold and cer_increase <= cer_threshold