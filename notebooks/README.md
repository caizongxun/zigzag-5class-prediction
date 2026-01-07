# Jupyter Notebooks Guide

Run notebooks in order. Each depends on previous output.

## 01_data_exploration.ipynb

**What it does:**
- Downloads BTC 15m data (219k candles)
- Shows price and volume charts
- Displays statistics
- Validates data quality

**Expected outputs:**
- BTC price chart over time
- Volume distribution
- Basic statistics (mean, std, min, max)

**Duration:** 2-3 minutes

---

## 02_feature_engineering.ipynb

**What it does:**
- Applies ZigZag indicator to label candles
- Generates 80+ technical indicators
- Creates features for model
- Shows label distribution

**Expected outputs:**
- Label distribution:
  - NO_SIGNAL: ~99.7%
  - HH: ~0.1%
  - LH: ~0.15%
  - HL: ~0.13%
  - LL: ~0.08%
- 75-87 features generated
- <0.1% missing values

**Duration:** 3-5 minutes

---

## 03_model_training.ipynb

**What it does:**
- Splits data into train/val/test
- Creates LSTM sequences
- Trains LSTM-XGBoost hybrid
- Evaluates on test set
- Trains volatility model

**Expected outputs:**
- LSTM Training accuracy: 70-75%
- LSTM Validation accuracy: 65-70%
- Test accuracy: 60-70%
- Volatility model F1: 0.75+
- Saved models in `./models/trained/`

**Duration:** 10-30 minutes (depends on CPU/GPU)

---

## Execution Instructions

### Option A: Sequential Execution

```bash
jupyter notebook
```

Then:
1. Open `01_data_exploration.ipynb`
2. Run all cells (Ctrl+A, Shift+Enter)
3. When complete, open `02_feature_engineering.ipynb`
4. Repeat for `03_model_training.ipynb`

### Option B: Run All at Once

In terminal, run one notebook at a time:

```bash
jupyter nbconvert --to notebook --execute 01_data_exploration.ipynb
jupyter nbconvert --to notebook --execute 02_feature_engineering.ipynb
jupyter nbconvert --to notebook --execute 03_model_training.ipynb
```

## Cell Breakdown

### Notebook 1: Data Exploration

**Cell 1:** Imports
- Standard libraries (pandas, numpy, matplotlib)
- Custom modules (CryptoDataFetcher, utils)

**Cell 2:** Download Data
- Fetches 219k BTC 15m candles
- ~8MB parquet file

**Cell 3:** Data Info
- Shows shape, columns, date range
- Memory usage

**Cell 4:** Visualization
- Price chart
- Volume bar chart

**Cell 5:** Statistics
- Mean, std, min, max prices
- Volume statistics

### Notebook 2: Feature Engineering

**Cell 1:** Imports
- ZigZagIndicator, FeatureEngineer
- Config utilities

**Cell 2:** Load Data
- Loads from cache if exists
- Or downloads fresh

**Cell 3:** Apply ZigZag
- Generates 5-class labels
- Shows distribution

**Cell 4:** Feature Engineering
- Computes 80+ indicators
- Takes 3-5 minutes
- Shows feature names

**Cell 5:** Data Quality
- Checks missing values
- Fills NaN with forward fill
- Validates feature values

### Notebook 3: Model Training

**Cell 1-2:** Setup
- Imports
- Loads data from previous notebook

**Cell 3:** Time Series Split
- Train: 70%
- Validation: 15%
- Test: 15%
- NO random shuffling (important!)

**Cell 4:** Sequence Creation
- Window size: 60 candles
- Prepares for LSTM
- Normalizes features

**Cell 5:** Train LSTM (Critical)
- Builds 4-layer neural net
- Trains 100 epochs (with early stopping)
- Shows progress bar
- Takes 10-30 minutes

**Cell 6:** Evaluate
- Tests on holdout set
- Shows accuracy, precision, recall, F1
- Confusion matrix

**Cell 7:** Volatility Model
- Trains separate classifier
- Predicts large moves vs ranging

**Cell 8:** Save Models
- Saves to `./models/trained/`
- Ready for inference

## Customization

Edit `config.yaml` to change:

```yaml
zigzag:
  depth: 12        # Smaller = more signals
  deviation: 5     # Larger = fewer signals

model:
  epochs: 100      # More = better but slower
  batch_size: 32   # Smaller = slower but more stable
```

## Common Issues

### Notebook Kernel Dies
- Reduce `batch_size` in cell 5
- Or use CPU only: `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'`

### Out of Memory
- Download smaller dataset
- Or use Google Colab (has GPU)

### Slow Training
- Normal on CPU (uses GPU if available)
- Can take 30+ minutes on CPU

### Low Accuracy
- Increase features
- Tune ZigZag parameters
- More training data

## Next Steps

1. After notebooks complete, models saved in `./models/trained/`
2. Use `python infer.py` for predictions
3. Deploy with `train.py` for production
4. See `DEPLOYMENT_GUIDE.md` for serving
