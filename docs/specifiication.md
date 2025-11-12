## Project Brief: Spotify Songs Trend Analysis & Recommender

**Dataset:** [Spotify Songs Dataset (Kaggle)](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)

**Goal:** Analyse music trends and build a recommendation engine.

### Scope & Constraints
- The dataset contains track-level metadata and audio features (e.g., `danceability`, `energy`, `valence`, `tempo`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `duration_ms`, `explicit`, `year`, etc.).
- It does **not** include listener-level demographics or user IDs. Therefore, trend analysis across demographics will use **proxies** (e.g., genre, year/decade, explicit flag, popularity) or be framed as a **stretch goal** requiring external data fusion.

### Phase 1 — ETL (Python Script)
- **Extract:** Load the CSV into pandas from a local path (placeholder `data/raw/spotify.csv`).
- **Transform:**
  - Standardise column names to `snake_case`.
  - Drop exact duplicates on a composite key (e.g., `track_name`, `artists`, `album_name`).
  - Parse/derive `release_year` (or `year`) and coerce to `Int64`.
  - Convert booleans (`explicit`) to `bool`.
  - Handle missing values (e.g., median for numeric features; drop rows missing all key audio features).
  - Optional outlier capping for features like `tempo`, `loudness`, `duration_ms` using quantile clipping (e.g., 0.5–99.5%).
- **Load:** Save as `data/clean/spotify_clean.csv` and persist a lightweight `parquet` copy for faster I/O.

**Functions to implement:**
```python
extract_data(path: str) -> pd.DataFrame
transform_data(df: pd.DataFrame) -> pd.DataFrame
load_data(df: pd.DataFrame, csv_path: str, pq_path: str) -> None
```

### Phase 2 — Feature Engineering (Python + Notebook with placeholders)
Implement a `feature_engineering.py` module with the following **placeholders** (documented with TODOs so juniors can fill them later):
```python
def encode_categorical_features(df):
    """TODO: Optional: one-hot encode `key`, `mode`, and derived `decade`. Keep reversible encodings for dashboards."""

def scale_numeric_features(df):
    """TODO: Standardise or min-max scale numeric features for distance-based models (retain original columns)."""

def create_derived_features(df):
    """TODO: Add `decade` from `year`; `tempo_bucket` (e.g., round to 5 or 10 BPM); `artist_count` from `artists`; \
    `energy_danceability` interaction; `mood_valence_energy` (e.g., harmonic mean)."""
```
**Notebook tasks:**
- EDA: distributions, correlations, feature importance heuristics (e.g., permutation on baseline model), top genres by `popularity`, trend lines by `year/decade`.
- Address class/imbalance issues only if you define a classification target (e.g., `popular = popularity >= 70`).

### Phase 3 — Modelling (Notebook)
Because user–item interactions are not present, favour **content-based** recommendations:
- **Primary:** Item–item similarity (cosine) using scaled audio features (`danceability`, `energy`, `valence`, `tempo`, `acousticness`, `instrumentalness`, `liveness`, `speechiness`, `loudness`).
- **Hybrid score:** Combine similarity with normalised `popularity` (e.g., `score = 0.8 * sim + 0.2 * pop_norm`).
- **Optional (stretch):** If playlist or user interaction data can be added, implement **implicit collaborative filtering** (e.g., ALS on a user–track matrix) and blend with content-based scores.

**Baseline predictive model (for LO alignment):**
- Train a simple model to predict `popularity` from audio features (e.g., `LinearRegression` or `RandomForestRegressor`) or a classifier for `popular` vs. `not_popular` (`LogisticRegression`, `DecisionTreeClassifier`).
- Evaluate using R²/MAE for regression or accuracy/F1/ROC-AUC for classification.

### Phase 4 — Dashboard (Streamlit → Gradio)
**Streamlit v1:**
- Tabs: **Overview**, **Trends**, **Recommender**.
- **Overview:** Dataset summary, missingness, feature glossary.
- **Trends:**
  - Filters: genre, decade, explicit flag.
  - Charts: popularity by genre & time, distribution of `tempo`, `valence`, and `energy` by genre.
- **Recommender:**
  - Text input/autocomplete for a seed track and artist.
  - Slider controls to weight similarity vs. popularity.
  - Table + audio preview URL (if present) for top-N recommendations.

**Gradio v2 (migration):**
- Replicate inputs/outputs with `gr.Interface` or `gr.TabbedInterface`.
- Provide a function `recommend(track_name, artist_name, k, sim_weight)` returning a dataframe of recommendations.

### Visualisations
- Popular songs by genre and time (bar, line with rolling mean by year/decade).
- Feature distributions per genre (box/violin).
- 2D projection (PCA or UMAP) of tracks coloured by genre; hover to reveal track/artist.

### Files & Deliverables
- `src/etl_pipeline.py`
- `src/feature_engineering.py`
- `notebooks/spotify_eda_and_models.ipynb`
- `app/dashboard_streamlit.py`
- `app/dashboard_gradio.py`
- `data/raw/` and `data/clean/` folders with README on data sources.

### Acceptance Criteria
- Reproducible ETL produces `spotify_clean.csv` with documented schema.
- Content-based recommender returns sensible top-N given a seed track.
- Streamlit app runs locally and shows at least three insights charts.
- Gradio app replicates the Recommender tab and one Trends chart.

### Mapping to Learning Outcomes
- **LO1–2:** EDA, feature prep, baseline model.
- **LO3–5:** ETL pipeline, cleaning, transformation.
- **LO6:** Brief section documenting ethical/data-governance considerations (e.g., biases in popularity; absence of demographics).
- **LO7:** Notebook section outlining research extensions (e.g., hybrid recommender with implicit feedback).
- **LO8:** Dashboard narratives and markdown tooltips.
- **LO9:** Discussion: applying similar pipeline to podcasts or audiobooks.
- **LO10–11:** README with maintenance plan, migration notes (Streamlit → Gradio), and lessons learned.
