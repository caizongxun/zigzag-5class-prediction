# ZigZag 5-Class Trading Signal Prediction

A machine learning system that predicts ZigZag indicator patterns (HH/HL/LH/LL) on BTC/crypto price data. Integrates pre-trained Stage 1 classifier from Hugging Face for bounce validation.

## Key Features

- **5-Class Classification**: HH (Higher High), HL (Higher Low), LH (Lower High), LL (Lower Low), NO_SIGNAL
- **80+ Technical Indicators**: Price action, momentum, volatility, volume, structure features
- **LSTM-XGBoost Hybrid**: Deep learning + gradient boosting for robust predictions
- **Hugging Face Integration**: Auto-download Stage 1 Bollinger Bands classifier
- **Production Ready**: Complete training pipeline, evaluation, and inference

## Data Source

OHLCV data from Hugging Face dataset:
- **Dataset**: `zongowo111/v2-crypto-ohlcv-data`
- **Timeframes**: 15m, 1h
- **Coverage**: Multiple crypto symbols with 5+ years of history
- **Size**: ~219k candles for BTC 15m

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/caizongxun/zigzag-5class-prediction
cd zigzag-5class-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Test Installation

```bash
python tests/test_zigzag.py
```

Expected output: All tests passed!

### 3. Download Data and Stage 1 Classifier

```bash
python data/fetch_data.py
```

This will:
- Download BTC 15m data from Hugging Face (219k candles)
- Download Stage 1 LSTM validity classifier

### 4. Run Jupyter Notebooks (In Order)

```bash
jupyter notebook
```

Execute notebooks:
1. `notebooks/01_data_exploration.ipynb` - Understand data distribution
2. `notebooks/02_feature_engineering.ipynb` - Generate 80+ features, apply ZigZag labels
3. `notebooks/03_model_training.ipynb` - Train LSTM and evaluate (10-30 minutes)

## Project Structure

```
zigzag-5class-prediction/
├── data/
│   ├── fetch_data.py              # Download from HF, Stage 1 classifier
│   └── raw/                       # Auto-created data cache
├── src/
│   ├── zigzag_indicator.py        # 5-class labeling logic
│   ├── features.py                # 80+ feature engineering
│   ├── models.py                  # LSTM-XGBoost architecture
│   ├── config.py                  # Configuration loader
│   └── utils.py                   # Utilities (split, normalize, save)
├── models/
│   ├── stage1/                    # Downloaded Stage 1 classifier
│   └── trained/                   # Your trained models
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── tests/
│   └── test_zigzag.py
├── config.yaml                    # Main configuration
├── requirements.txt
└── README.md
```

## Signal Definitions

| Signal | Meaning | Action | Confidence |
|--------|---------|--------|------------|
| **HH** | Higher High | Short opportunity | Trend strengthening |
| **HL** | Higher Low | Long opportunity | Trend continuing |
| **LH** | Lower High | Short opportunity | Pressure increasing |
| **LL** | Lower Low | Short opportunity | Trend weakening |
| **NO_SIGNAL** | No clear pattern | Wait | Ambiguous structure |

## Model Architecture

### LSTM Encoder
- 2 LSTM layers (128, 64 units)
- Dropout 0.3 for regularization
- Input: 60 timesteps x 86 features
- Output: 5-class probabilities

### Training
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Epochs: 100 with early stopping
- Validation split: 20%

### Expected Performance
- Training accuracy: 70-75%
- Validation accuracy: 65-70%
- Test accuracy: 60-70%

## Using Stage 1 Classifier

The system automatically downloads the pre-trained Bollinger Bands bounce validity classifier:

```python
from data.fetch_data import CryptoDataFetcher

# Auto-download on first run
model, scaler, config = CryptoDataFetcher.load_stage1_model(
    model_type='lstm_validity',
    model_dir='./models/stage1'
)

# Use for validation
predictions = model.predict(X_normalized)
valid_bounces = predictions > 0.5
```

## Configuration

Edit `config.yaml` to customize:

```yaml
zigzag:
  depth: 12          # Pivot lookback
  deviation: 5       # Min % change for reversal
  backstep: 2        # Filter small moves

model:
  lstm:
    layers: [128, 64]
    dropout: 0.3
  epochs: 100
  batch_size: 32
```

## Performance Optimization

If accuracy is below target:

1. **Increase features**: Adjust `lookback_periods` in config
2. **Tune LSTM**: Add layers or increase units in `src/models.py`
3. **Adjust ZigZag**: Modify `depth`, `deviation` parameters
4. **More data**: Train on longer history (adjust `train_ratio`)
5. **Balance classes**: Use `class_weight` in training

## Inference Example

```python
from data.fetch_data import CryptoDataFetcher
from src.zigzag_indicator import ZigZagIndicator
from src.features import FeatureEngineer
from src.models import LSTMXGBoostModel

# Load data
fetcher = CryptoDataFetcher()
btc_data = fetcher.fetch_symbol_timeframe('BTCUSDT', '15m')

# Apply ZigZag
zigzag = ZigZagIndicator()
btc_data = zigzag.label_kbars(btc_data)

# Engineer features
fe = FeatureEngineer()
btc_data = fe.calculate_all_features(btc_data)

# Load model and predict
model = LSTMXGBoostModel()
model.load('./models/trained')

feature_cols = fe.get_feature_columns(btc_data)
X = btc_data[feature_cols].values[-60:].reshape(1, 60, -1)

class_pred, confidence = model.predict(X)
signal_name = ZigZagIndicator.get_label_name(class_pred[0])

print(f'Signal: {signal_name}, Confidence: {confidence[0]:.2%}')
```

## Troubleshooting

### "Module not found" Error

```bash
pip install -r requirements.txt --upgrade
```

### CUDA/GPU Issues

To use CPU only:

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### Slow Data Download

First download is slow (~5min). Subsequent runs use cache.

### Model Training Stuck

Check GPU memory: `nvidia-smi` or reduce `batch_size` in config.

## Contributing

Feel free to:
- Add more features
- Optimize model architecture
- Test on other timeframes
- Implement ensemble methods

## License

MIT License

## References

- ZigZag Indicator: Classic price structure analysis
- LSTM for time series: Hochreiter & Schmidhuber (1997)
- LightGBM: Microsoft Research
- Hugging Face: Open-source ML platform

## Citation

If you use this project:

```bibtex
@software{zigzag5class2026,
  title={ZigZag 5-Class Trading Signal Prediction},
  author={Zong},
  year={2026},
  url={https://github.com/caizongxun/zigzag-5class-prediction}
}
```

## Support

For issues or questions, open an issue on GitHub.
