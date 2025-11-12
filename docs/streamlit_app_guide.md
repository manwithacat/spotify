# Streamlit Dashboard User Guide

## 🎵 Spotify Track Analytics Dashboard

An interactive web application for exploring Spotify track data and predicting track popularity using machine learning.

---

## Quick Start

### Launch the Dashboard

```bash
# Activate virtual environment
source .venv/bin/activate

# Run Streamlit app
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

---

## Dashboard Features

### 📊 Tab 1: Data Explorer

**Purpose**: Browse and filter the Spotify tracks dataset

**Features**:
- **Quick Stats**: Total tracks, average popularity, unique genres and artists
- **Interactive Filters**:
  - Multi-select genre filter
  - Popularity range slider
- **Data Table**: View first 100 filtered tracks with key attributes
- **Download**: Export filtered data as CSV

**Use Cases**:
- Explore tracks by specific genres
- Find all high-popularity tracks (70-100)
- Download subsets for further analysis

---

### 📈 Tab 2: Visualizations

**Purpose**: Rich, interactive visualizations of dataset characteristics

**Visualizations Include**:

1. **Popularity Distribution** - Histogram showing track popularity spread
2. **Audio Feature Distributions** - 6 histograms for key features
3. **Top 20 Genres** - Horizontal bar chart of most common genres
4. **Mood & Energy Classification** - Pie charts showing track categorization
5. **Feature Correlation Heatmap** - Identify relationships between features
6. **Interactive Scatter Plot** - Custom X/Y axis selection with color coding

**Interactive Elements**:
- Hover over data points for details
- Zoom, pan, and download charts
- Select axes and color schemes for scatter plots

**Use Cases**:
- Identify feature correlations
- Understand genre distribution
- Explore mood patterns in music

---

### 🤖 Tab 3: ML Model

**Purpose**: Understand the machine learning model's performance and feature importance

**Features**:

1. **Model Metrics**:
   - Model Type: XGBoost Regressor
   - Test R² Score: 0.3882
   - Test RMSE: 17.38

2. **Feature Importance Chart**:
   - Top 20 features ranked by importance
   - Genre is the strongest predictor
   - Track length and explicit content are significant

3. **Performance Visualizations**:
   - Predictions vs Actual scatter plot
   - Residual distribution (error analysis)

4. **Model Metadata**:
   - Expandable JSON view of complete model specs
   - Hyperparameters, training info, performance metrics

**Use Cases**:
- Understand what drives popularity predictions
- Evaluate model accuracy
- Export model specifications

---

### 🎯 Tab 4: Track Predictor (Interactive UX)

**Purpose**: Predict popularity for new tracks and get optimization recommendations

#### Input Methods

**1. 🎚️ Sliders (Beginner-Friendly)**
- Interactive sliders for all track characteristics
- Tooltips explain each feature
- Default values pre-set

**2. 📝 Manual Entry (Advanced)**
- Same as sliders, but for precise control
- Best for users who know exact target values

**3. 🎲 Random Example**
- Load a random track from the dataset
- Great for exploring how real tracks score
- See actual track name and artist

#### Track Characteristics Input

**Audio Features (0-1 scale)**:
- 🕺 **Danceability**: How suitable for dancing
- ⚡ **Energy**: Intensity and activity level
- 😊 **Valence**: Musical positiveness (happy vs sad)
- 🎸 **Acousticness**: Likelihood of being acoustic
- 🎹 **Instrumentalness**: Probability of no vocals
- 🗣️ **Speechiness**: Presence of spoken words
- 🎤 **Liveness**: Likelihood of live performance

**Other Features**:
- 🔊 **Loudness**: -60 to 0 dB
- 🥁 **Tempo**: 50-200 BPM
- ⏱️ **Duration**: 1-10 minutes
- 🔞 **Explicit Content**: Yes/No
- 🎹 **Key**: C, C#, D, ... B
- 🎵 **Mode**: Major/Minor
- ⏰ **Time Signature**: 3/4, 4/4, 5/4
- 🎸 **Genre**: Select from 114 genres

#### Prediction Output

After clicking **🚀 Predict Popularity**:

1. **Prediction Score** (0-100)
   - Displayed in large green box
   - Color-coded interpretation:
     - 70-100: Excellent viral potential 🎉
     - 50-69: Good, needs targeted promotion 👍
     - 30-49: Moderate, needs optimization ⚠️
     - 0-29: Challenging, refine characteristics 📉

2. **AI-Powered Recommendations** 💡
   - Up to 5 specific, actionable recommendations
   - Each recommendation includes:
     - **Factor**: What to change (e.g., Danceability)
     - **Current vs Suggested**: Where you are vs target
     - **Potential Impact**: Expected popularity gain
     - **Tip**: How to implement the change

3. **Similar Successful Tracks** 🎵
   - Shows top 5 popular tracks in same genre
   - Compare your settings to successful songs
   - Learn from what works in your genre

---

## Use Case: Music Producer Workflow

### Scenario: You're producing a new pop track

1. **Explore Market** (Tab 2):
   - Check popularity distribution for pop genre
   - Identify common characteristics of successful pop

2. **Understand Drivers** (Tab 3):
   - Review feature importance
   - Note that genre, energy, and danceability matter most

3. **Predict Your Track** (Tab 4):
   - Input your track's audio characteristics
   - Get initial popularity prediction

4. **Optimize** (Tab 4 Recommendations):
   - Review AI suggestions
   - Example: "Increase danceability from 0.5 to 0.7 for +10 points"
   - Adjust your production accordingly

5. **Iterate**:
   - Update sliders with new values
   - Re-predict to see improvement
   - Fine-tune until satisfied

6. **Compare** (Tab 4 Similar Tracks):
   - Check against successful pop tracks
   - Ensure your characteristics align with hits

---

## Example Recommendations Explained

### 🕺 Danceability Recommendation
```
Current: 0.45 → Suggested: 0.7+
Potential Impact: +8-12 popularity points
Tip: Add a stronger, more rhythmic beat
```

**Action**: Increase the groove, add percussion, make the rhythm more prominent

### ⚡ Energy Recommendation
```
Current: 0.50 → Suggested: 0.7+
Potential Impact: +5-10 popularity points
Tip: Boost intensity with louder instruments
```

**Action**: Turn up the mix, add more dynamic range, increase tempo

### 😊 Positivity Recommendation
```
Current: 0.30 → Suggested: 0.5-0.7
Potential Impact: +5-8 popularity points
Tip: Try major key or uplifting melodies
```

**Action**: Shift from minor to major key, add upbeat chord progressions

---

## Technical Details

### Data Filters Performance
- Filters 114,000 tracks in real-time
- Responsive multi-select and range controls
- Download filtered data up to 100K rows

### Visualization Performance
- Plots are cached for fast loading
- Scatter plots sample 5,000 points for speed
- All charts are interactive (zoom, pan, hover)

### Model Prediction
- Instant predictions (<100ms)
- Handles all 37 engineered features automatically
- Clips predictions to valid 0-100 range

### Data Caching
- Dataset loaded once and cached
- Model loaded once for entire session
- Filters don't reload data

---

## Tips for Best Experience

1. **Start with Random Example**: Get familiar with how real tracks score
2. **Use Sliders**: More intuitive than manual entry
3. **Iterate Quickly**: Adjust one feature at a time to see impact
4. **Compare Genres**: Different genres have different success patterns
5. **Follow Recommendations**: AI tips are based on successful tracks
6. **Download Data**: Export filtered datasets for offline analysis

---

## Troubleshooting

### App Won't Start
```bash
# Check if Streamlit is installed
pip list | grep streamlit

# Reinstall if needed
pip install streamlit

# Ensure you're in project root
cd /path/to/spotify_track_analytics_popularity_prediction
streamlit run app.py
```

### Prediction Errors
- Ensure all sliders are moved from default
- Check that genre is selected
- Try "Random Example" to test with known-good data

### Slow Performance
- Close other browser tabs
- Reduce number of filtered tracks
- Use smaller date ranges for visualizations

---

## Deployment Options

### Local Network Access

Allow others on your network to access:
```bash
streamlit run app.py --server.address=0.0.0.0
```

### Cloud Deployment

**Streamlit Community Cloud** (Free):
1. Push to GitHub
2. Visit share.streamlit.io
3. Connect repository
4. Deploy automatically

**Heroku**:
- Already configured with `Procfile` and `setup.sh`
- Push to Heroku repository
- App auto-deploys

---

## Future Enhancements

- [ ] SHAP values for prediction explanations
- [ ] A/B testing between track variations
- [ ] Genre-specific recommendations
- [ ] Batch prediction from CSV upload
- [ ] Historical popularity trends
- [ ] Collaborative filtering ("similar to" search)

---

**App Version**: 1.0
**Last Updated**: 2025-11-12
**Feedback**: Report issues via GitHub
