# Spotify Track Analytics - Project Structure

## Directory Layout

```
spotify_track_analytics_popularity_prediction/
├── .claude/                    # Claude Code configuration
│   └── CLAUDE.md              # Project guidance for Claude Code
├── assets/                     # Static assets (images, logos)
│   └── teamlogo.png
├── data/                       # Data directory (files not in git)
│   ├── raw/                   # Raw immutable data from Kaggle
│   │   └── dataset.csv        # 114k Spotify tracks dataset
│   ├── processed/             # Cleaned and processed datasets
│   └── README.md              # Data documentation
├── docs/                       # Project documentation
├── jupyter_notebooks/          # Legacy notebooks (credit card analysis)
├── notebooks/                  # Main analysis notebooks
│   └── Hackathon2Music.ipynb  # Primary music analytics pipeline
├── outputs/                    # Generated outputs (not in git)
│   ├── eda/                   # Exploratory data analysis results
│   ├── models/                # Trained models
│   └── plots/                 # Generated visualizations
├── src/                        # Python source modules
│   └── README.md              # Source code guidelines
├── .venv/                      # Python virtual environment
├── .gitignore                 # Git ignore rules
├── Procfile                   # Heroku deployment config
├── README.md                  # Project overview
├── requirements.txt           # Python dependencies
└── setup.sh                   # Streamlit configuration script
```

## Quick Start

### 1. Environment Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Setup

Data is automatically excluded from git. To get started:

```bash
cd data/raw
kaggle datasets download -d maharshipandya/-spotify-tracks-dataset
unzip -o ./-spotify-tracks-dataset.zip -d .
rm ./-spotify-tracks-dataset.zip
```

### 3. Run Analysis

```bash
# Launch Jupyter for interactive analysis
jupyter notebook notebooks/Hackathon2Music.ipynb

# Or run Streamlit app (when implemented)
streamlit run app.py
```

## Key Features

- **114,000+ tracks** across 125 genres
- **21 audio features** including danceability, energy, valence, tempo
- **Mood/Energy classification** system
- **Genre-level analysis** and visualizations
- **Popularity prediction** foundation

## Data Flow

1. **Raw data** → `data/raw/dataset.csv`
2. **Processing** → Cleaning, feature engineering, standardization
3. **Analysis** → EDA, correlations, hypothesis testing
4. **Outputs** → Cleaned data to `data/processed/`, visuals to `outputs/`

## Important Notes

- All data files are in `.gitignore` (only data/README.md is tracked)
- Output directories are created automatically by notebooks
- Legacy credit card analysis exists in `jupyter_notebooks/` but is unrelated to this project
- Project is configured for Heroku deployment with Streamlit
