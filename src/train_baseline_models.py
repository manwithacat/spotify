"""
Train Baseline Models (Linear Regression & Random Forest) on Cleaned Dataset

This script trains baseline models for comparison with XGBoost on the
cleaned, deduplicated dataset (78,310 tracks).

Models trained:
1. Linear Regression - Simple linear model
2. Random Forest - Ensemble tree-based model

Purpose: Establish baseline performance for comparison with XGBoost
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🎵 TRAINING BASELINE MODELS ON CLEANED DATASET")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# STEP 1: Load Cleaned Dataset
# ============================================================================
print("="*80)
print("📂 STEP 1: LOAD CLEANED DATASET")
print("="*80)

df = pd.read_parquet('data/processed/cleaned_spotify_data.parquet')
print(f"✓ Loaded {len(df):,} tracks")
print(f"✓ Columns: {list(df.columns)}")
print()

# ============================================================================
# STEP 2: Prepare Features
# ============================================================================
print("="*80)
print("🎯 STEP 2: PREPARE FEATURES")
print("="*80)

# Use same 9 audio features as XGBoost model
feature_cols = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

X = df[feature_cols].copy()
y = df['popularity'].copy()

# Remove NaN values
mask = ~(X.isnull().any(axis=1) | y.isnull())
X, y = X[mask], y[mask]

print(f"✓ Features: {feature_cols}")
print(f"✓ Target: popularity")
print(f"✓ Samples: {len(X):,}")
print(f"✓ Features shape: {X.shape}")
print(f"✓ Target range: {y.min():.1f} - {y.max():.1f} (mean: {y.mean():.1f})")
print()

# ============================================================================
# STEP 3: Train/Test Split
# ============================================================================
print("="*80)
print("✂️  STEP 3: TRAIN/TEST SPLIT")
print("="*80)

# Use same split as XGBoost for fair comparison: 70% train, 15% val, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"✓ Train: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"✓ Val: {X_val.shape[0]:,} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"✓ Test: {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
print()

# ============================================================================
# STEP 4: Train Linear Regression
# ============================================================================
print("="*80)
print("📊 STEP 4: TRAIN LINEAR REGRESSION")
print("="*80)

lr_model = LinearRegression()
print("Training Linear Regression...")
lr_model.fit(X_train, y_train)
print("✓ Training complete")
print()

# Evaluate Linear Regression
def evaluate_model(model, X, y, name=""):
    """Calculate metrics for a dataset"""
    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    # Adjusted R²
    n = len(y)
    p = X.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    print(f"{name} Metrics:")
    print(f"  R² = {r2:.4f}")
    print(f"  Adjusted R² = {adj_r2:.4f}")
    print(f"  RMSE = {rmse:.2f}")
    print(f"  MAE = {mae:.2f}")
    print()

    return {
        'r2': r2,
        'adj_r2': adj_r2,
        'rmse': rmse,
        'mae': mae
    }

lr_train_metrics = evaluate_model(lr_model, X_train, y_train, "Linear Regression - Training")
lr_val_metrics = evaluate_model(lr_model, X_val, y_val, "Linear Regression - Validation")
lr_test_metrics = evaluate_model(lr_model, X_test, y_test, "Linear Regression - Test")

# ============================================================================
# STEP 5: Train Random Forest
# ============================================================================
print("="*80)
print("🌲 STEP 5: TRAIN RANDOM FOREST")
print("="*80)

# Use default hyperparameters for baseline
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

print("Training Random Forest (100 trees)...")
rf_model.fit(X_train, y_train)
print("✓ Training complete")
print()

rf_train_metrics = evaluate_model(rf_model, X_train, y_train, "Random Forest - Training")
rf_val_metrics = evaluate_model(rf_model, X_val, y_val, "Random Forest - Validation")
rf_test_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest - Test")

# ============================================================================
# STEP 6: Compare with XGBoost
# ============================================================================
print("="*80)
print("📈 STEP 6: MODEL COMPARISON")
print("="*80)

# XGBoost metrics from previous training
xgb_test_r2 = 0.1619
xgb_test_rmse = 16.32
xgb_test_mae = 13.14

comparison_data = {
    'Model': ['Linear Regression', 'Random Forest', 'XGBoost (tuned)'],
    'Test R²': [lr_test_metrics['r2'], rf_test_metrics['r2'], xgb_test_r2],
    'Test RMSE': [lr_test_metrics['rmse'], rf_test_metrics['rmse'], xgb_test_rmse],
    'Test MAE': [lr_test_metrics['mae'], rf_test_metrics['mae'], xgb_test_mae]
}

comparison_df = pd.DataFrame(comparison_data)
print("Model Performance Comparison:")
print(comparison_df.to_string(index=False))
print()

# Determine best model
best_r2_idx = comparison_df['Test R²'].idxmax()
best_model_name = comparison_df.loc[best_r2_idx, 'Model']
print(f"🏆 Best Model: {best_model_name} (R² = {comparison_df.loc[best_r2_idx, 'Test R²']:.4f})")
print()

# ============================================================================
# STEP 7: Save Results
# ============================================================================
print("="*80)
print("💾 STEP 7: SAVE RESULTS")
print("="*80)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save comparison results
results = {
    'timestamp': datetime.now().isoformat(),
    'dataset': 'cleaned_spotify_data_v2',
    'n_samples': len(df),
    'n_features': len(feature_cols),
    'feature_names': feature_cols,
    'data_split': {
        'train_size': X_train.shape[0],
        'val_size': X_val.shape[0],
        'test_size': X_test.shape[0]
    },
    'models': {
        'linear_regression': {
            'train': lr_train_metrics,
            'val': lr_val_metrics,
            'test': lr_test_metrics
        },
        'random_forest': {
            'hyperparameters': {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            },
            'train': rf_train_metrics,
            'val': rf_val_metrics,
            'test': rf_test_metrics
        },
        'xgboost_tuned': {
            'note': 'From previous training run',
            'test': {
                'r2': xgb_test_r2,
                'rmse': xgb_test_rmse,
                'mae': xgb_test_mae
            }
        }
    },
    'comparison': comparison_df.to_dict(orient='records'),
    'best_model': best_model_name
}

# Save to JSON
output_path = Path(f'outputs/metadata/baseline_models_comparison_{timestamp}.json')
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✓ Results saved: {output_path}")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*80)
print("📊 BASELINE MODELS SUMMARY")
print("="*80)

print(f"""
Dataset: 78,310 cleaned tracks (deduplicated, zero-popularity removed)
Features: 9 audio features (same as XGBoost)

LINEAR REGRESSION:
  Test R²:   {lr_test_metrics['r2']:.4f}
  Test RMSE: {lr_test_metrics['rmse']:.2f}
  Test MAE:  {lr_test_metrics['mae']:.2f}

RANDOM FOREST (100 trees):
  Test R²:   {rf_test_metrics['r2']:.4f}
  Test RMSE: {rf_test_metrics['rmse']:.2f}
  Test MAE:  {rf_test_metrics['mae']:.2f}

XGBOOST (tuned):
  Test R²:   {xgb_test_r2:.4f}
  Test RMSE: {xgb_test_rmse:.2f}
  Test MAE:  {xgb_test_mae:.2f}

BEST MODEL: {best_model_name}

Key Insights:
- Simple linear regression shows R² = {lr_test_metrics['r2']:.4f} (baseline)
- Random Forest achieves R² = {rf_test_metrics['r2']:.4f} (tree-based baseline)
- XGBoost achieves R² = {xgb_test_r2:.4f} (tuned model)
- Improvement: XGBoost outperforms Linear Regression by {((xgb_test_r2 - lr_test_metrics['r2']) / lr_test_metrics['r2'] * 100):.1f}%
- Improvement: XGBoost outperforms Random Forest by {((xgb_test_r2 - rf_test_metrics['r2']) / rf_test_metrics['r2'] * 100):.1f}%

All models limited by audio-only features (R² ~0.16 ceiling).
Artist features expected to improve R² to 0.28-0.32.
""")

print("="*80)
print(f"✅ BASELINE TRAINING COMPLETE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
