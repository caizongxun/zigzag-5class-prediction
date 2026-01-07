# Quick Start Guide - ZigZag 5-Class Predictor

## 5 Minutes Setup

### 1. Install

```bash
git clone https://github.com/caizongxun/zigzag-5class-prediction
cd zigzag-5class-prediction

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Verify

```bash
python tests/test_zigzag.py
# Output: All tests passed!
```

### 3. Download Data

```bash
python data/fetch_data.py
```

### 4. Train

**Interactive (Jupyter):**
```bash
jupyter notebook
# Run: 01_data_exploration.ipynb -> 02_feature_engineering.ipynb -> 03_model_training.ipynb
```

**Batch (Command line):**
```bash
python train.py --epochs 100 --batch_size 32
```

### 5. Predict

```bash
python infer.py --model ./models/trained --symbol BTCUSDT --timeframe 15m
```

## Key Signals

- **HH**: Higher High - Short when pressure increases
- **HL**: Higher Low - Long when trend continues up
- **LH**: Lower High - Short when trend continues down  
- **LL**: Lower Low - Short when selling accelerates
- **NO_SIGNAL**: Wait for clarity

## Expected Results

- Test Accuracy: 60-70%
- Training Time: 10-30 minutes
- Inference Time: <10ms per prediction

## Troubleshooting

### "ModuleNotFoundError: No module named 'tensorflow'"

```bash
pip install -r requirements.txt --upgrade
```

### "CUDA not found" (OK, uses CPU)

Model works on CPU, just slower.

### Data download slow

First download ~5 minutes. Cached after that.

## Next Steps

1. Understand signals: Read `README.md`
2. Tune model: Edit `config.yaml`
3. Deploy: See `DEPLOYMENT_GUIDE.md`
4. Monitor: Track accuracy over time

## Support

Issue? Check GitHub: https://github.com/caizongxun/zigzag-5class-prediction
