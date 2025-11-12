# Data Directory

This directory contains the Spotify Tracks Dataset used for popularity prediction analysis.

## Structure

```
data/
├── raw/          # Raw, immutable data from Kaggle
│   └── dataset.csv
└── processed/    # Cleaned and processed datasets
```

## Dataset Information

**Source**: [Kaggle - Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)

**Size**: ~114,000 tracks across 125 genres

**Columns** (21 features):
- `track_id`: Unique Spotify identifier
- `artists`: Artist name(s)
- `album_name`: Album title
- `track_name`: Song title
- `popularity`: Score 0-100 based on play counts
- `duration_ms`: Track length in milliseconds
- `explicit`: Boolean for explicit content
- `danceability`: 0.0-1.0 dancing suitability score
- `energy`: 0.0-1.0 intensity/loudness score
- `key`: Musical key (0=C, 1=C♯/D♭, etc.)
- `loudness`: Volume in decibels
- `mode`: Scale type (1=major, 0=minor)
- `speechiness`: Spoken word content estimate
- `acousticness`: 0.0-1.0 acoustic likelihood
- `instrumentalness`: Probability of no vocals
- `liveness`: Live recording likelihood
- `valence`: 0.0-1.0 positivity (sad to happy)
- `tempo`: Speed in BPM
- `time_signature`: Musical meter (e.g., 4 = 4/4)
- `track_genre`: Genre classification

## Usage

### Download Fresh Data

```bash
# From project root with Kaggle API credentials configured
cd data/raw
kaggle datasets download -d maharshipandya/-spotify-tracks-dataset
unzip -o ./-spotify-tracks-dataset.zip -d .
rm ./-spotify-tracks-dataset.zip
```

### Load Data in Python

```python
import pandas as pd

# Load raw data
df = pd.read_csv('data/raw/dataset.csv')

# Load processed data (after running analysis pipeline)
df_clean = pd.read_csv('data/processed/cleaned_spotify_data.csv')
```

## Notes

- Raw data files are **not tracked in git** (see `.gitignore`)
- This README.md is the only tracked file in the data directory
- Always work with copies of raw data; never modify `raw/` contents directly
