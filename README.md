# Whisper Speed Optimization Analysis

A comprehensive analysis of OpenAI Whisper transcription quality vs audio speed tradeoffs.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete analysis
python analyze.py

# Collect new dataset (optional)
python collect_data.py
```

## Results Summary

- **Optimal Speed**: 1.5x (33% cost savings, 2.4% quality loss)
- **Conservative**: 1.2x-1.3x (17-23% savings, <5% quality loss)  
- **Aggressive**: 1.7x-1.8x (41-44% savings, ~5.5% quality loss)

## Repository Structure

```
whisper_speed_optimizer/
├── src/                    # Core modules
│   ├── speed_analyzer.py   # Main analysis logic
│   ├── visualizer.py       # Plotting and charts
│   ├── audio_processor.py  # Audio speed manipulation
│   └── whisper_client.py   # OpenAI API interface
├── data/                   # Dataset
├── baseline_1x_analysis/   # Core results
├── plots/                  # All visualizations
├── analyze.py             # Main analysis script
└── collect_data.py        # Dataset collection
```

## Key Findings

1. **1.5x speed is optimal** - Best balance of cost and quality
2. **Content-dependent results** - Educational content more tolerant
3. **Quality cliff at 2.0x** - Significant degradation beyond 2x
4. **0.1x increments reveal nuances** - Smooth optimization curve

## Analysis Data

- **Videos**: 19 educational YouTube videos
- **Speeds**: 1.0x to 2.0x in 0.1x increments (209 experiments)
- **Method**: 1.0x baseline comparison using WER metrics
- **API**: OpenAI Whisper-1 model

Generated from comprehensive 209-experiment analysis validating the [original blog hypothesis](https://george.mand.is/2025/06/openai-charges-by-the-minute-so-make-the-minutes-shorter/).
