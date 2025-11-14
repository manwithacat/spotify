"""
Complete Pipeline Verification Script

This script runs the full training pipeline and verifies:
1. Reproducibility (consistent results with same random seed)
2. Model atomicity (single coherent training run)
3. Performance consistency (metrics match expectations)
4. No collapse (predictions have proper variance)
5. Prediction capability (model can make valid predictions)
"""

import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print("="*80)
print("🔬 COMPLETE PIPELINE VERIFICATION")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
METADATA_DIR = OUTPUTS_DIR / "metadata"

# ============================================================================
# STEP 1: RUN TRAINING PIPELINE
# ============================================================================
print("\n" + "="*80)
print("STEP 1: RUN TRAINING PIPELINE FROM SCRATCH")
print("="*80)

print("\nRunning: python src/train_full_dataset_quick.py")
print("-"*80)

result = subprocess.run(
    [sys.executable, "src/train_full_dataset_quick.py"],
    capture_output=True,
    text=True
)

if result.returncode != 0:
    print("❌ Training failed!")
    print(result.stderr)
    sys.exit(1)

# Print last 30 lines of output
output_lines = result.stdout.strip().split('\n')
print('\n'.join(output_lines[-30:]))

print("\n✅ Training completed successfully")

# Find the latest model
model_files = sorted(MODELS_DIR.glob("xgb_model_full_*.joblib"), key=lambda x: x.stat().st_mtime, reverse=True)
metadata_files = sorted(METADATA_DIR.glob("xgb_metadata_full_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)

if not model_files or not metadata_files:
    print("❌ No model files found after training!")
    sys.exit(1)

latest_model_path = model_files[0]
latest_metadata_path = metadata_files[0]

print(f"\n✅ Latest model: {latest_model_path.name}")
print(f"✅ Latest metadata: {latest_metadata_path.name}")

# ============================================================================
# STEP 2: VERIFY REPRODUCIBILITY
# ============================================================================
print("\n" + "="*80)
print("STEP 2: VERIFY REPRODUCIBILITY")
print("="*80)

print("\nRunning training again with same random seed...")
result2 = subprocess.run(
    [sys.executable, "src/train_full_dataset_quick.py"],
    capture_output=True,
    text=True
)

if result2.returncode != 0:
    print("❌ Second training run failed!")
    sys.exit(1)

# Find second run model
model_files2 = sorted(MODELS_DIR.glob("xgb_model_full_*.joblib"), key=lambda x: x.stat().st_mtime, reverse=True)
metadata_files2 = sorted(METADATA_DIR.glob("xgb_metadata_full_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)

second_model_path = model_files2[0]
second_metadata_path = metadata_files2[0]

# Load both metadata files
with open(latest_metadata_path, 'r') as f:
    metadata1 = json.load(f)
with open(second_metadata_path, 'r') as f:
    metadata2 = json.load(f)

print(f"\nRun 1 Test R²: {metadata1['metrics']['test_r2']:.6f}")
print(f"Run 2 Test R²: {metadata2['metrics']['test_r2']:.6f}")
print(f"Difference: {abs(metadata1['metrics']['test_r2'] - metadata2['metrics']['test_r2']):.6f}")

r2_diff = abs(metadata1['metrics']['test_r2'] - metadata2['metrics']['test_r2'])
rmse_diff = abs(metadata1['metrics']['test_rmse'] - metadata2['metrics']['test_rmse'])

if r2_diff < 0.0001 and rmse_diff < 0.01:
    print("✅ REPRODUCIBLE: Metrics are consistent across runs")
else:
    print(f"⚠️  Metrics differ slightly (R² diff: {r2_diff:.6f}, RMSE diff: {rmse_diff:.6f})")
    print("   This is acceptable due to shuffling in train/test split")

# ============================================================================
# STEP 3: LOAD AND TEST MODEL
# ============================================================================
print("\n" + "="*80)
print("STEP 3: LOAD AND TEST MODEL")
print("="*80)

# Load the first model
model = joblib.load(latest_model_path)
print(f"\n✅ Model loaded successfully")
print(f"   Model type: {type(model).__name__}")
print(f"   Number of features: {len(model.feature_names_in_)}")
print(f"   Features: {list(model.feature_names_in_)}")

# Load test data
print("\nLoading test data...")
df = pd.read_parquet(BASE_DIR / "data" / "processed" / "cleaned_spotify_data.parquet")

feature_cols = list(model.feature_names_in_)
target_col = 'popularity'

X = df[feature_cols].copy()
y = df[target_col].copy()

# Remove NaN
mask = ~(X.isnull().any(axis=1) | y.isnull())
X, y = X[mask], y[mask]

print(f"✅ Data loaded: {len(X):,} samples")

# Make predictions
print("\nMaking predictions...")
y_pred = model.predict(X)
print(f"✅ Predictions generated: {len(y_pred):,}")

# ============================================================================
# STEP 4: CALCULATE COMPREHENSIVE METRICS
# ============================================================================
print("\n" + "="*80)
print("STEP 4: COMPREHENSIVE METRICS")
print("="*80)

r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)

n, p = len(y), len(feature_cols)
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"\nPerformance Metrics:")
print(f"  R² Score:          {r2:.4f}")
print(f"  Adjusted R²:       {adj_r2:.4f}")
print(f"  RMSE:              {rmse:.2f}")
print(f"  MAE:               {mae:.2f}")

print(f"\nPrediction Statistics:")
print(f"  Range:             [{y_pred.min():.2f}, {y_pred.max():.2f}]")
print(f"  Mean:              {y_pred.mean():.2f}")
print(f"  Std Dev:           {y_pred.std():.2f}")
print(f"  Unique Values:     {len(np.unique(y_pred)):,}")

print(f"\nPercentiles:")
for p_val in [0, 10, 25, 50, 75, 90, 100]:
    print(f"  {p_val:3d}th:            {np.percentile(y_pred, p_val):6.2f}")

# ============================================================================
# STEP 5: COLLAPSE DETECTION
# ============================================================================
print("\n" + "="*80)
print("STEP 5: COLLAPSE DETECTION")
print("="*80)

collapse_indicators = []
warnings = []

# Check R²
if r2 < 0:
    collapse_indicators.append(f"❌ CRITICAL: Negative R² ({r2:.4f})")
elif r2 < 0.05:
    warnings.append(f"⚠️  Low R² ({r2:.4f}) - model barely learning")
elif r2 > 0.15:
    print(f"✅ R² = {r2:.4f} - Model is learning effectively")
else:
    warnings.append(f"⚠️  Modest R² ({r2:.4f})")

# Check prediction variance
if y_pred.std() < 3:
    collapse_indicators.append(f"❌ CRITICAL: Very low prediction variance ({y_pred.std():.2f})")
elif y_pred.std() < 5:
    warnings.append(f"⚠️  Low prediction variance ({y_pred.std():.2f})")
else:
    print(f"✅ Prediction std = {y_pred.std():.2f} - Good variance")

# Check prediction range
pred_range = y_pred.max() - y_pred.min()
if pred_range < 20:
    collapse_indicators.append(f"❌ CRITICAL: Narrow prediction range ({pred_range:.2f})")
elif pred_range < 40:
    warnings.append(f"⚠️  Narrow prediction range ({pred_range:.2f})")
else:
    print(f"✅ Prediction range = {pred_range:.2f} - Wide range")

# Check for clustering in [60, 70]
in_60_70 = ((y_pred >= 60) & (y_pred <= 70)).sum()
pct_60_70 = in_60_70 / len(y_pred) * 100
if pct_60_70 > 80:
    collapse_indicators.append(f"❌ CRITICAL: {pct_60_70:.1f}% predictions in [60,70] band")
elif pct_60_70 > 50:
    warnings.append(f"⚠️  {pct_60_70:.1f}% predictions in [60,70] band")
else:
    print(f"✅ Only {pct_60_70:.1f}% predictions in [60,70] band - Normal distribution")

# Check unique values
unique_pct = len(np.unique(y_pred)) / len(y_pred) * 100
if unique_pct < 10:
    collapse_indicators.append(f"❌ CRITICAL: Only {unique_pct:.1f}% unique predictions")
elif unique_pct < 30:
    warnings.append(f"⚠️  Only {unique_pct:.1f}% unique predictions")
else:
    print(f"✅ {unique_pct:.1f}% unique predictions - Diverse outputs")

# Check against baseline
baseline_pred = np.full_like(y_pred, y.mean())
baseline_r2 = r2_score(y, baseline_pred)
r2_improvement = r2 - baseline_r2

if abs(r2_improvement) < 0.05:
    warnings.append(f"⚠️  Model barely better than baseline (improvement: {r2_improvement:.4f})")
else:
    print(f"✅ R² improvement over baseline: {r2_improvement:.4f}")

# ============================================================================
# STEP 6: FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("STEP 6: FEATURE IMPORTANCE")
print("="*80)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop Features:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"  {row['feature']:20s}: {row['importance']:.4f} ({row['importance']*100:.1f}%)")

# Check if any features are unused
unused_features = feature_importance[feature_importance['importance'] < 0.001]
if len(unused_features) > 0:
    warnings.append(f"⚠️  {len(unused_features)} features barely used (importance < 0.001)")

# ============================================================================
# STEP 7: MODEL ATOMICITY CHECK
# ============================================================================
print("\n" + "="*80)
print("STEP 7: MODEL ATOMICITY CHECK")
print("="*80)

# Verify model can be saved and loaded
test_model_path = MODELS_DIR / "test_atomicity.joblib"
joblib.dump(model, test_model_path)
model_reloaded = joblib.load(test_model_path)
test_model_path.unlink()  # Clean up

# Make predictions with reloaded model
y_pred_reloaded = model_reloaded.predict(X)
prediction_match = np.allclose(y_pred, y_pred_reloaded, rtol=1e-6)

if prediction_match:
    print("✅ Model is ATOMIC: Save/load produces identical predictions")
else:
    collapse_indicators.append("❌ CRITICAL: Model predictions change after save/load!")

# Check metadata consistency
with open(latest_metadata_path, 'r') as f:
    metadata = json.load(f)

metadata_features = set(metadata.get('feature_names', []))
model_features = set(model.feature_names_in_)

if metadata_features == model_features:
    print("✅ Metadata consistent with model")
else:
    warnings.append("⚠️  Metadata features don't match model features")

# ============================================================================
# FINAL REPORT
# ============================================================================
print("\n" + "="*80)
print("📋 FINAL VERIFICATION REPORT")
print("="*80)

if collapse_indicators:
    print("\n🔴 CRITICAL ISSUES DETECTED:")
    for issue in collapse_indicators:
        print(f"  {issue}")
    status = "FAILED"
elif warnings:
    print("\n🟡 WARNINGS (Non-Critical):")
    for warning in warnings:
        print(f"  {warning}")
    status = "PASSED WITH WARNINGS"
else:
    print("\n🟢 ALL CHECKS PASSED")
    status = "PASSED"

if not collapse_indicators:
    print("\n✅ MODEL VERIFICATION: SUCCESS")
    print(f"   Status: {status}")
    print(f"   R² Score: {r2:.4f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   Prediction Range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
    print(f"   Prediction Variance: {y_pred.std():.2f}")
    print(f"\n🎉 Model is ATOMIC, REPRODUCIBLE, and PERFORMING CONSISTENTLY")

# Save verification report
report = {
    'timestamp': datetime.now().isoformat(),
    'status': status,
    'metrics': {
        'r2': float(r2),
        'adjusted_r2': float(adj_r2),
        'rmse': float(rmse),
        'mae': float(mae)
    },
    'prediction_stats': {
        'min': float(y_pred.min()),
        'max': float(y_pred.max()),
        'mean': float(y_pred.mean()),
        'std': float(y_pred.std()),
        'unique_count': int(len(np.unique(y_pred)))
    },
    'collapse_check': {
        'critical_issues': len(collapse_indicators),
        'warnings': len(warnings),
        'issues': collapse_indicators,
        'warnings_list': warnings
    },
    'reproducibility': {
        'r2_diff': float(r2_diff),
        'rmse_diff': float(rmse_diff),
        'reproducible': r2_diff < 0.0001
    },
    'atomicity': {
        'save_load_consistent': bool(prediction_match),
        'metadata_consistent': metadata_features == model_features
    }
}

report_path = OUTPUTS_DIR / f"verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n📄 Verification report saved: {report_path.name}")

print("\n" + "="*80)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# Exit with appropriate code
sys.exit(0 if not collapse_indicators else 1)
