#!/usr/bin/env python3
"""
Main script for Whisper Speed Optimization Analysis.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from speed_analyzer import SpeedAnalyzer
from visualizer import Visualizer

def main():
    """Main entry point for speed analysis."""
    
    print("🚀 Whisper Speed Optimization Analysis")
    print("======================================")
    
    # Initialize analyzer
    analyzer = SpeedAnalyzer()
    
    # Run complete analysis
    print("📊 Running speed analysis...")
    results = analyzer.analyze_all_speeds()
    
    # Generate visualizations
    print("📈 Creating visualizations...")
    visualizer = Visualizer()
    visualizer.create_all_plots(results)
    
    print("✅ Analysis complete! Check plots/ directory for results.")

if __name__ == "__main__":
    main()
