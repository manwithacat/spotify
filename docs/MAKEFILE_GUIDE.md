# Makefile Command Reference

Complete guide to using the Makefile for development, testing, and deployment.

---

## Quick Start

```bash
# Show all available commands
make help

# Install everything and run
make quick-start
```

---

## Core Commands

### Development Workflow

#### `make install`
Install all dependencies including development tools.

**Installs**:
- Production dependencies from `requirements.txt`
- Development tools: black, flake8, pylint, nbdime, papermill
- Jupyter kernel: `spotify_env`
- Git integration for notebook diffs

**Usage**:
```bash
make install
```

#### `make dashboard`
Launch the Streamlit web application.

**Opens**: http://localhost:8501
**Stop**: Press `Ctrl+C`

**Usage**:
```bash
make dashboard
```

**Features**:
- 📊 Data Explorer
- 📈 Visualizations
- 🤖 ML Model Analysis
- 🎯 Track Predictor

---

## Pipeline Execution

### `make pipeline`
Run the complete ML pipeline: ETL → Feature Engineering → ML Training.

**Steps**:
1. ETL Pipeline (2-3 seconds)
2. Feature Engineering (2-3 seconds)
3. Model Training (<1 second)

**Output**:
- `data/processed/cleaned_spotify_data.parquet`
- `data/processed/X_train.parquet`, `X_test.parquet`
- `outputs/models/xgboost_popularity_model.joblib`

**Usage**:
```bash
make pipeline
```

### Individual Pipeline Steps

#### `make etl`
Run only the ETL (Extract, Transform, Load) pipeline.

**Input**: `data/raw/dataset.csv`
**Output**:
- `data/processed/cleaned_spotify_data.csv`
- `data/processed/cleaned_spotify_data.parquet`

#### `make fe`
Run only Feature Engineering.

**Input**: `data/processed/cleaned_spotify_data.parquet`
**Output**:
- Train/test splits
- ML-ready features (37 features)

#### `make train`
Train the XGBoost model only.

**Input**: `data/processed/X_train.parquet`, `y_train.parquet`
**Output**:
- `outputs/models/xgboost_popularity_model.joblib`
- `outputs/models/model_metadata.json`

---

## Code Quality

### Linting

#### `make lint`
Run fast linting with flake8 (recommended for development).

**Checks**:
- Code style (PEP 8)
- Common errors
- Line length (120 chars)

**Usage**:
```bash
make lint
```

**Example Output**:
```
🔍 Running flake8 linting...
app.py:14:1: F401 'pathlib.Path' imported but unused
src/etl_pipeline.py:253:9: F841 local variable unused
```

#### `make lint-strict`
Run comprehensive linting with pylint (slower, more thorough).

**Checks**:
- All flake8 checks
- Code complexity
- Documentation
- Naming conventions

**Usage**:
```bash
make lint-strict
```

### Code Formatting

#### `make format`
Automatically format all Python code with Black.

**Changes**:
- Applies PEP 8 style
- Fixes indentation
- Normalizes quotes
- Line length: 120 chars

**Usage**:
```bash
make format
```

**Example**:
```bash
$ make format
✨ Formatting code with black...
reformatted src/etl_pipeline.py
reformatted app.py
All done! ✨ 🍰 ✨
```

#### `make format-check`
Check if code is formatted (doesn't modify files).

**Usage**:
```bash
make format-check
```

Use in CI/CD to ensure code is formatted before merge.

---

## Testing

### `make test`
Run the complete pipeline verification test suite.

**Tests**:
- ✅ Data files exist and are readable
- ✅ Model files exist
- ✅ Model loads successfully
- ✅ Predictions work
- ✅ Pipeline scripts present
- ✅ Notebooks exist

**Usage**:
```bash
make test
```

**Example Output**:
```
🧪 Running tests...
✅ PASS     Data Files
✅ PASS     Model Files
✅ PASS     Model Loading
✅ PASS     Pipeline Scripts
✅ PASS     Notebooks
```

---

## Notebook Execution

### Papermill Integration

#### `make nb-etl`
Run ETL notebook with papermill (automated execution).

**Input**: `notebooks/00_ETL_Pipeline.ipynb`
**Output**: `notebooks/00_ETL_Pipeline_output.ipynb`

#### `make nb-fe`
Run Feature Engineering notebook.

**Input**: `notebooks/02_Feature_Engineering.ipynb`
**Output**: `notebooks/02_Feature_Engineering_output.ipynb`

#### `make nb-ml`
Run ML training notebook.

**Input**: `notebooks/03_ML_XGBoost_Model.ipynb`
**Output**: `notebooks/03_ML_XGBoost_Model_output.ipynb`

#### `make nb-all`
Run all notebooks in sequence.

**Usage**:
```bash
make nb-all
```

### Notebook Utilities

#### `make nbconvert`
Convert notebooks to Python scripts.

**Output**: `.py` files alongside `.ipynb` files

**Usage**:
```bash
make nbconvert
```

#### `make nbdiff`
Open nbdime web interface for visual notebook diffs.

**Features**:
- Side-by-side comparison
- Ignores cell outputs
- Shows structural changes
- Better than `git diff` for notebooks

**Usage**:
```bash
make nbdiff
```

---

## Maintenance

### Cleanup Commands

#### `make clean`
Remove temporary files and caches.

**Removes**:
- `__pycache__` directories
- `.pyc` files
- `.pytest_cache`
- Notebook output files (`*_output.ipynb`)

**Usage**:
```bash
make clean
```

**Safe**: Doesn't delete data or models

#### `make clean-data`
Remove all processed data (keeps raw data).

**⚠️ Warning**: Interactive confirmation required

**Removes**:
- `data/processed/*`

**Preserves**:
- `data/raw/*`

**Usage**:
```bash
make clean-data
```

#### `make clean-models`
Remove trained models.

**⚠️ Warning**: Interactive confirmation required

**Removes**:
- `outputs/models/*`

**Usage**:
```bash
make clean-models
```

### Dependency Management

#### `make requirements`
Generate frozen requirements file.

**Output**: `requirements_frozen.txt` with exact versions

**Usage**:
```bash
make requirements
```

#### `make dev`
Install development dependencies.

**Installs**:
- black, flake8, pylint
- pytest, nbdime
- papermill, jupyter

**Usage**:
```bash
make dev
```

---

## Status & Information

### `make status`
Show current pipeline status.

**Displays**:
- ✅/❌ Raw data present
- ✅/❌ Cleaned data present
- ✅/❌ Training data present
- ✅/❌ Model trained
- ✅/❌ Dashboard ready

**Usage**:
```bash
make status
```

**Example Output**:
```
📊 Pipeline Status
==================

Data Files:
  ✅ Raw data present
  ✅ Cleaned data present
  ✅ Training data present

Models:
  ✅ Model trained

App:
  ✅ Dashboard ready
```

---

## Workflows

### Complete Workflow

#### `make all`
Run full workflow: clean → pipeline → test.

**Steps**:
1. Clean temporary files
2. Run ETL
3. Run Feature Engineering
4. Train model
5. Run verification tests

**Usage**:
```bash
make all
```

**Time**: ~10 seconds

#### `make quick-start`
Quick start: install → pipeline → dashboard.

**Steps**:
1. Install dependencies
2. Run complete pipeline
3. Launch dashboard

**Usage**:
```bash
# First time setup
make quick-start
```

**Time**: ~30 seconds (first run)

---

## Common Workflows

### Daily Development

```bash
# Start day: check status
make status

# Make code changes
# ...

# Format code
make format

# Check linting
make lint

# Test changes
make test

# Launch dashboard
make dashboard
```

### Before Commit

```bash
# Format code
make format

# Run linting
make lint

# Run tests
make test

# Clean up
make clean
```

### Full Rebuild

```bash
# Clean everything
make clean-data
make clean-models

# Rebuild from scratch
make pipeline

# Verify
make test
make status
```

### Production Deployment

```bash
# Install dependencies
make install

# Run pipeline
make pipeline

# Verify
make test

# Launch
make dashboard
```

---

## Integration with Git

### Notebook Diffs

nbdime is configured globally for better notebook diffs.

**Git diff for notebooks**:
```bash
git diff notebook.ipynb
# Shows semantic diff (ignores outputs)
```

**Visual diff**:
```bash
nbdiff notebook_v1.ipynb notebook_v2.ipynb
# Opens in browser
```

**Git merge conflicts**:
```bash
nbdime mergetool
# Interactive conflict resolution
```

---

## Troubleshooting

### Command Not Found

**Problem**: `make: command not found`

**Solution**:
```bash
# macOS
xcode-select --install

# Linux
sudo apt-get install make
```

### Permission Denied

**Problem**: `Permission denied` when running Makefile

**Solution**:
```bash
chmod +x Makefile
```

### Module Not Found

**Problem**: `ModuleNotFoundError` when running commands

**Solution**:
```bash
# Reinstall dependencies
make install

# Or activate venv first
source .venv/bin/activate
make pipeline
```

### Streamlit Won't Start

**Problem**: Dashboard won't launch

**Solution**:
```bash
# Check if port is in use
lsof -ti:8501 | xargs kill -9

# Try again
make dashboard
```

---

## Advanced Usage

### Custom Parameters

Override default parameters:

```bash
# Use different Python
make PYTHON=python3.11 pipeline

# Use different Streamlit port
streamlit run app.py --server.port=8502
```

### Parallel Execution

Run pipeline steps in parallel (if independent):

```bash
# NOT recommended (steps depend on each other)
make -j4 etl fe train  # ❌ Will fail

# Recommended: run sequentially
make pipeline  # ✅ Correct
```

### Dry Run

See what commands will run without executing:

```bash
make -n pipeline
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Pipeline CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.12'
      - run: make install
      - run: make lint
      - run: make format-check
      - run: make test
```

---

## Performance Tips

1. **Use `make status`** before running pipeline (check if already done)
2. **Use `make lint`** instead of `make lint-strict` for faster feedback
3. **Run `make clean`** periodically to free up space
4. **Use `make test`** to verify without re-running pipeline

---

## Summary of Most Used Commands

| Command | Purpose | Frequency |
|---------|---------|-----------|
| `make help` | Show all commands | First time |
| `make dashboard` | Launch app | Daily |
| `make pipeline` | Run ML pipeline | After data changes |
| `make test` | Verify pipeline | Before commit |
| `make lint` | Check code quality | Before commit |
| `make format` | Auto-format code | Before commit |
| `make status` | Check what's done | Anytime |
| `make clean` | Cleanup temp files | Weekly |

---

**Makefile Version**: 1.0
**Last Updated**: 2025-11-12
**Targets**: 27 commands
