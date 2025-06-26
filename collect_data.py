#!/usr/bin/env python3
"""
Dataset collection and preparation script.
"""

import sys
from pathlib import Path

# Add src to Python path  
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dataset_manager import DatasetManager

def main():
    """Main entry point for dataset management."""
    
    print("📥 Dataset Collection and Preparation")
    print("====================================")
    
    # Initialize dataset manager
    manager = DatasetManager()
    
    # Collect dataset
    print("🎥 Collecting video dataset...")
    manager.collect_videos()
    
    # Process audio
    print("🎵 Processing audio files...")
    manager.process_audio()
    
    print("✅ Dataset ready!")

if __name__ == "__main__":
    main()
