# Gradio Dashboard Guide

## Overview

The Gradio dashboard (`app_gradio.py`) provides an alternative interface to the Streamlit dashboard with the same core functionality. Gradio offers a simpler, more lightweight interface that's ideal for quick prototyping and sharing.

## 🚀 Quick Start

### Launch the Dashboard

```bash
# Using make command (recommended)
make dashboard-gradio

# Or run directly
python app_gradio.py
```

The dashboard will launch at **http://localhost:7860**

## 📋 Features

The Gradio dashboard provides 4 main tabs with identical functionality to the Streamlit version:

### 1. 📊 Data Explorer Tab

**Purpose:** Browse and filter the dataset of 114,000 Spotify tracks

**Features:**
- View dataset statistics (total tracks, avg popularity, genres, artists)
- Filter by genre (multi-select dropdown)
- Filter by popularity range (dual sliders)
- View filtered data in interactive table
- Download filtered data as CSV

**Usage:**
1. Select one or more genres from the dropdown (optional)
2. Adjust the popularity range sliders
3. Click "Apply Filters" to update the table
4. Browse the top 100 matching tracks

### 2. 📈 Visualizations Tab

**Purpose:** Explore data through interactive charts

**Features:**
- **Popularity Distribution**: Histogram showing how popularity is distributed across all tracks
- **Audio Features Distribution**: 6 histograms showing distributions of danceability, energy, valence, acousticness, instrumentalness, and speechiness
- **Top Genres by Count**: Top 20 genres ranked by number of tracks
- **Top Genres by Popularity**: Top 20 genres ranked by average popularity score
- **Energy vs Valence**: Scatter plot colored by popularity (5,000 sample)
- **Feature Correlations**: Heatmap showing correlations between numeric features

**Usage:**
- Click on any accordion to expand/collapse visualizations
- All charts are interactive (hover for details, zoom, pan)
- Charts use Plotly for rich interactivity

### 3. 🤖 ML Model Tab

**Purpose:** Understand the machine learning model's performance and behavior

**Features:**
- **Model Information Panel**:
  - Performance metrics (R², RMSE, MAE)
  - Training details (sample sizes, features)
  - Hyperparameters (estimators, depth, learning rate)

- **Feature Importance Chart**: Top 20 features ranked by importance
- **Predictions Scatter Plot**: Actual vs predicted popularity on test set
  - Red diagonal line shows perfect predictions
  - Points closer to line = better predictions

**Model Metrics:**
- R² Score: 0.3882 (explains 39% of variance)
- RMSE: 17.38
- MAE: 13.00

### 4. 🎯 Track Predictor Tab

**Purpose:** Predict popularity for new tracks and get optimization recommendations

**Input Groups:**

**Audio Features:**
- **Danceability** (0-1): How suitable for dancing
- **Energy** (0-1): Intensity and activity level
- **Valence** (0-1): Musical positivity (happy vs sad)
- **Acousticness** (0-1): Acoustic vs electronic
- **Instrumentalness** (0-1): Amount of vocals
- **Speechiness** (0-1): Presence of spoken words
- **Liveness** (0-1): Presence of audience
- **Loudness** (-60 to 0 dB): Overall loudness

**Musical Properties:**
- **Key** (0-11): Musical key (C=0, C#=1, ..., B=11)
- **Mode** (0/1): Minor (0) or Major (1)
- **Time Signature** (1/3/4/5): Beats per bar

**Track Details:**
- **Tempo** (40-220 BPM): Beats per minute
- **Duration** (30,000-600,000 ms): Track length in milliseconds
- **Genre**: Select from 114 available genres

**Usage:**

**Method 1: Load Random Example**
1. Click "🎲 Load Random Example" button
2. A random track from the dataset will populate all fields
3. Click "🎯 Predict Popularity" to see results

**Method 2: Manual Input**
1. Adjust all sliders and dropdowns to your desired values
2. Click "🎯 Predict Popularity"
3. View the prediction and recommendations

**Output:**
- **Predicted Popularity Score**: 0-100 scale
- **Track Characteristics**: Mood, energy level, tempo category, duration
- **Optimization Recommendations** (up to 5):
  - Which factor to improve
  - Current value
  - Suggested target
  - Estimated impact on popularity
  - Actionable tip

**Example Recommendations:**
- "Increase danceability to 0.7+ for +8-12 popularity points"
- "Add vocals to reduce instrumentalness for +7-12 points"
- "Use major keys to brighten valence for +5-8 points"

## 🎨 Interface Differences from Streamlit

### Advantages of Gradio:
- **Simpler setup**: Single `demo.launch()` call
- **Built-in sharing**: Easy to create public links with `share=True`
- **Mobile-friendly**: Better responsive design out of the box
- **Queue support**: Built-in request queuing for high traffic
- **API mode**: Can be used as REST API with `api_name` parameter

### Advantages of Streamlit:
- **More customization**: Better CSS control and theming
- **Richer widgets**: More component types
- **Better caching**: More sophisticated caching decorators
- **Session state**: More powerful state management

## 🔧 Configuration

The app can be configured in the `demo.launch()` call:

```python
demo.launch(
    server_name="0.0.0.0",  # Listen on all network interfaces
    server_port=7860,        # Port number
    share=False,             # Set True to create public link
    show_error=True          # Show detailed errors
)
```

**Common configurations:**

**Development (local only):**
```python
demo.launch(server_name="127.0.0.1", server_port=7860)
```

**Production (with public link):**
```python
demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
```

**Custom domain:**
```python
demo.launch(server_name="0.0.0.0", server_port=80, share=False)
```

## 📊 Performance

- **Load time**: ~5-10 seconds (loads all data and model at startup)
- **Prediction time**: <100ms per prediction
- **Visualization rendering**: 1-2 seconds for complex plots
- **Memory usage**: ~500MB (same as Streamlit)

**Optimization tips:**
- Data is loaded once at startup (not cached per request)
- Plotly charts are generated on-demand when tabs are opened
- Sample datasets for large scatter plots (5,000 points max)

## 🐛 Troubleshooting

### App won't start

**Error: "ModuleNotFoundError: No module named 'gradio'"**
```bash
pip install gradio>=4.0.0
```

**Error: "Address already in use"**
```bash
# Kill existing process
pkill -f "app_gradio.py"

# Or change port in app
demo.launch(server_port=7861)
```

### Predictions seem wrong

**Issue:** Feature engineering mismatch
- The predictor uses simplified feature engineering
- For production use, integrate with `src/feature_engineering.py`

**Solution:**
```python
from src.feature_engineering import FeatureEngineer
fe = FeatureEngineer(feature_vector)
processed = fe.run()
prediction = model.predict(processed)
```

### Charts not displaying

**Issue:** Plotly not installed
```bash
pip install plotly>=5.0.0
```

**Issue:** JavaScript blocked in browser
- Check browser console for errors
- Disable ad blockers/security extensions
- Try different browser

### Slow performance

**Solutions:**
- Reduce sample sizes in scatter plots
- Use `gr.Accordion(open=False)` for heavy charts
- Implement lazy loading for visualizations
- Add `queue()` before `launch()` for queuing

## 🚀 Deployment Options

### 1. Hugging Face Spaces (Free)

```bash
# Create a new Space on huggingface.co
# Upload:
# - app_gradio.py
# - requirements.txt
# - data/ and outputs/ directories
```

**Requirements.txt for Spaces:**
```
gradio>=4.0.0
pandas
plotly
joblib
scikit-learn
xgboost
pyarrow
```

### 2. Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python", "app_gradio.py"]
```

**Build and run:**
```bash
docker build -t spotify-gradio .
docker run -p 7860:7860 spotify-gradio
```

### 3. Cloud Platforms

**Google Cloud Run:**
```bash
gcloud run deploy spotify-gradio \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

**AWS Elastic Beanstalk:**
```bash
eb init -p python-3.11 spotify-gradio
eb create spotify-gradio-env
eb deploy
```

## 📝 Comparison: Gradio vs Streamlit

| Feature | Gradio | Streamlit |
|---------|--------|-----------|
| **Setup Complexity** | ⭐⭐⭐⭐⭐ Very Easy | ⭐⭐⭐⭐ Easy |
| **Customization** | ⭐⭐⭐ Moderate | ⭐⭐⭐⭐⭐ Extensive |
| **Performance** | ⭐⭐⭐⭐ Good | ⭐⭐⭐⭐ Good |
| **Mobile Support** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐ Fair |
| **Sharing** | ⭐⭐⭐⭐⭐ Built-in | ⭐⭐⭐ Requires Cloud |
| **API Support** | ⭐⭐⭐⭐⭐ Native | ⭐⭐ Third-party |
| **Documentation** | ⭐⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Excellent |
| **Community** | ⭐⭐⭐⭐ Large | ⭐⭐⭐⭐⭐ Very Large |

**Use Gradio when:**
- You want quick prototyping
- You need easy sharing (public links)
- You want mobile-first design
- You need API access
- You're deploying to Hugging Face Spaces

**Use Streamlit when:**
- You need extensive customization
- You want richer UI components
- You need complex state management
- You want better caching control
- You're building a production dashboard

## 📚 Additional Resources

- **Gradio Documentation**: https://gradio.app/docs
- **Gradio Guides**: https://gradio.app/guides
- **Hugging Face Spaces**: https://huggingface.co/spaces
- **Plotly Documentation**: https://plotly.com/python/

## 🤝 Contributing

To modify the Gradio dashboard:

1. Edit `app_gradio.py`
2. Test locally: `make dashboard-gradio`
3. Update this guide if adding features
4. Run linting: `make lint`
5. Format code: `make format`

## 📄 License

Same as parent project - see main README.md

---

**Last Updated**: 2025-11-12
**Version**: 1.0
**Compatibility**: Gradio >=4.0.0
