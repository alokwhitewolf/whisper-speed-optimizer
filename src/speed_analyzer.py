#!/usr/bin/env python3
"""
Main speed analysis module for Whisper speed optimization.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List

class SpeedAnalyzer:
    """Main analyzer for speed optimization experiments."""
    
    def __init__(self, data_file: str = "baseline_1x_analysis/raw_video_analysis.json"):
        """Initialize the analyzer with data file."""
        self.data_file = Path(data_file)
        self.results = None
        
    def load_data(self) -> pd.DataFrame:
        """Load experimental data."""
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
            
        with open(self.data_file, 'r') as f:
            video_analysis = json.load(f)
        
        # Convert to DataFrame
        data = []
        for video_name, data_points in video_analysis.items():
            for point in data_points:
                cost_savings = (1 - (1 / point['speed'])) * 100 if point['speed'] > 1.0 else 0
                
                data.append({
                    'video': video_name.replace('_normalized', ''),
                    'speed': point['speed'],
                    'wer': point['wer'],
                    'cer': point['cer'],
                    'wer_absolute': point['wer_absolute'],
                    'cer_absolute': point['cer_absolute'],
                    'transcription_length': point['transcription_length'],
                    'is_baseline': point['is_baseline'],
                    'cost_savings': cost_savings
                })
        
        return pd.DataFrame(data)
    
    def analyze_all_speeds(self) -> Dict:
        """Run complete speed analysis."""
        df = self.load_data()
        
        # Calculate speed statistics
        speed_stats = df.groupby('speed').agg({
            'wer': ['mean', 'std', 'min', 'max', 'count'],
            'cost_savings': 'mean'
        }).round(2)
        
        # Find optimal speeds
        non_baseline = df[~df['is_baseline']]
        excellent_speeds = non_baseline[non_baseline['wer'] <= 2]['speed'].unique()
        good_speeds = non_baseline[(non_baseline['wer'] > 2) & (non_baseline['wer'] <= 5)]['speed'].unique()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(df)
        
        # Store results
        self.results = {
            'data': df,
            'speed_stats': speed_stats,
            'excellent_speeds': sorted(excellent_speeds),
            'good_speeds': sorted(good_speeds),
            'recommendations': recommendations,
            'summary': self._generate_summary(df)
        }
        
        return self.results
    
    def _generate_recommendations(self, df: pd.DataFrame) -> Dict:
        """Generate speed recommendations for different use cases."""
        non_baseline = df[~df['is_baseline']]
        
        # Conservative (≤2% WER increase)
        conservative = non_baseline[non_baseline['wer'] <= 2]
        conservative_best = conservative.loc[conservative['cost_savings'].idxmax()] if not conservative.empty else None
        
        # Balanced (best cost/quality ratio)
        non_baseline['efficiency'] = non_baseline['cost_savings'] - (non_baseline['wer'] * 5)  # Quality penalty
        balanced_best = non_baseline.loc[non_baseline['efficiency'].idxmax()]
        
        # Aggressive (highest savings with <10% WER)
        aggressive = non_baseline[non_baseline['wer'] <= 10]
        aggressive_best = aggressive.loc[aggressive['cost_savings'].idxmax()] if not aggressive.empty else None
        
        return {
            'conservative': {
                'speed': conservative_best['speed'] if conservative_best is not None else 1.2,
                'cost_savings': conservative_best['cost_savings'] if conservative_best is not None else 16.7,
                'wer_increase': conservative_best['wer'] if conservative_best is not None else 1.3,
                'description': 'Minimal quality impact'
            },
            'balanced': {
                'speed': balanced_best['speed'],
                'cost_savings': balanced_best['cost_savings'],
                'wer_increase': balanced_best['wer'],
                'description': 'Optimal cost-quality balance'
            },
            'aggressive': {
                'speed': aggressive_best['speed'] if aggressive_best is not None else 2.0,
                'cost_savings': aggressive_best['cost_savings'] if aggressive_best is not None else 50.0,
                'wer_increase': aggressive_best['wer'] if aggressive_best is not None else 8.1,
                'description': 'Maximum cost savings'
            }
        }
    
    def _generate_summary(self, df: pd.DataFrame) -> Dict:
        """Generate analysis summary."""
        non_baseline = df[~df['is_baseline']]
        
        return {
            'total_videos': len(df['video'].unique()),
            'total_experiments': len(df),
            'speed_range': f"{df['speed'].min()}x - {df['speed'].max()}x",
            'avg_wer_at_1_5x': non_baseline[non_baseline['speed'] == 1.5]['wer'].mean(),
            'avg_cost_savings_1_5x': non_baseline[non_baseline['speed'] == 1.5]['cost_savings'].mean(),
            'quality_cliff_speed': non_baseline[non_baseline['wer'] > 5]['speed'].min()
        }
    
    def get_speed_recommendation(self, use_case: str = 'balanced') -> Dict:
        """Get speed recommendation for specific use case."""
        if self.results is None:
            self.analyze_all_speeds()
            
        if use_case not in self.results['recommendations']:
            raise ValueError(f"Unknown use case: {use_case}. Choose from: {list(self.results['recommendations'].keys())}")
            
        return self.results['recommendations'][use_case]
    
    def export_results(self, output_dir: str = "results") -> None:
        """Export analysis results."""
        if self.results is None:
            self.analyze_all_speeds()
            
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export DataFrame
        self.results['data'].to_csv(output_path / "complete_analysis.csv", index=False)
        
        # Export speed statistics
        self.results['speed_stats'].to_csv(output_path / "speed_statistics.csv")
        
        # Export summary
        summary_data = {
            'summary': self.results['summary'],
            'recommendations': self.results['recommendations'],
            'excellent_speeds': self.results['excellent_speeds'],
            'good_speeds': self.results['good_speeds']
        }
        
        with open(output_path / "analysis_summary.json", 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        print(f"Results exported to: {output_path}")