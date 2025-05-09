# AI Coach - Basketball Shot Analysis

An AI-powered basketball shot analysis system that compares user shots with professional players' form.

## Project Structure

```
aiCoach/
├── src/                    # Source code
│   ├── analysis/          # Analysis-related modules
│   │   ├── shot_analyzer.py
│   │   ├── shot_comparator.py
│   │   ├── analyze_findings.py
│   │   └── pose_analyzer.py
│   ├── utils/             # Utility functions
│   │   ├── video_downloader.py
│   │   ├── video_collector.py
│   │   └── scanner.py
│   └── visualization/     # Visualization-related code
├── data/                  # Data directories
│   ├── raw/              # Raw videos
│   │   ├── nba_videos/
│   │   └── users_shots/
│   ├── processed/        # Processed data
│   │   └── shot_data/
│   └── results/          # Analysis results
│       ├── analysis_results/
│       ├── comparison_results/
│       └── analysis_findings/
├── bin/                   # Executable scripts
│   ├── run_shot_analysis.py
│   ├── compare_user_shot.py
│   └── process_videos.py
├── tests/                 # Test files
├── docs/                  # Documentation
└── requirements.txt       # Project dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your shot videos in `data/raw/users_shots/`
2. Run the analysis:
```bash
python bin/run_shot_analysis.py
```

## Features

- Pose detection and analysis
- Shot form comparison with professional players
- Detailed feedback and recommendations
- Visualization of shot mechanics

## Dependencies

- Python 3.8+
- OpenCV
- MediaPipe
- NumPy
- Matplotlib
- scikit-learn 