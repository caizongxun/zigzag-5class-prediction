# Troubleshooting Guide

## Installation Issues

### 1. Jupyter Lab Extension Installation Error

**Error Message:**
```
OSError: [Errno 2] No such file or directory: '...jupyterlab-manager...js.map'
```

**Cause:** Jupyter Lab extension path is too long on Windows or permissions issue.

**Solutions (in order of ease):**

#### Quick Fix: Use Lightweight Notebook
```bash
# Use notebook instead of jupyterlab
pip uninstall jupyter jupyterlab -y
pip install notebook==7.0.0 ipykernel==6.26.0
```

#### Full Fix: Clean Virtual Environment
```bash
# Windows
deactivate
rmdir /s /q .venv
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt --no-cache-dir
```

#### Using Conda (Most Stable)
```bash
conda create -n zigzag python=3.10
conda activate zigzag
conda install -c conda-forge jupyter jupyterlab tensorflow keras
pip install -r requirements.txt
```

---

### 2. TensorFlow Installation Hangs

**Problem:** TensorFlow download takes very long or times out.

**Solution:**
```bash
# Use cached installation
pip install tensorflow==2.14.0 --prefer-binary

# Or use CPU-only version
pip install tensorflow-cpu==2.14.0
```

---

### 3. Module Not Found Errors

**Error:** `ModuleNotFoundError: No module named 'tensorflow'`

**Causes & Fixes:**

1. Virtual environment not activated:
   ```bash
   # Windows
   .\venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

2. Wrong Python interpreter:
   ```bash
   # Check which Python is running
   which python
   python --version  # Should be 3.10+
   
   # Reinstall requirements
   pip install -r requirements.txt
   ```

3. Pip cache corrupted:
   ```bash
   pip install --upgrade --force-reinstall -r requirements.txt
   ```

---

### 4. CUDA/GPU Not Found

**Error:** `Could not load dynamic library 'cudart64_110.dll'`

**This is OK** - TensorFlow will use CPU instead (slower but works).

To silence warning:
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings
import tensorflow as tf
```

**To use GPU:**
- Install CUDA 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive
- Install cuDNN 8.6: https://developer.nvidia.com/cudnn
- Follow installation guide

---

## Runtime Issues

### 5. Out of Memory Error

**Error:** `ResourceExhaustedError: OOM when allocating tensor`

**Solutions:**

1. Reduce batch size in config:
   ```yaml
   model:
     batch_size: 16  # Default: 32
   ```

2. Use CPU only:
   ```bash
   # Before running train.py
   set CUDA_VISIBLE_DEVICES=-1  # Windows
   export CUDA_VISIBLE_DEVICES=-1  # Linux
   ```

3. Reduce dataset size:
   ```python
   # In fetch_data.py
   n_candles = 50000  # Default: 219k
   ```

4. Clear TensorFlow session:
   ```python
   import tensorflow as tf
   tf.keras.backend.clear_session()
   ```

---

### 6. Training Very Slow

**Problem:** Training takes 30+ minutes on normal machine.

**This is expected on CPU.** TensorFlow uses CPU by default.

**Speed comparisons:**
- CPU (no GPU): 30-60 minutes
- NVIDIA GPU: 2-5 minutes
- Apple M1/M2: 5-10 minutes

**To check if GPU is used:**
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

**To use GPU, install:**
- Windows: CUDA 11.8 + cuDNN 8.6
- Mac: TensorFlow Metal (built-in)
- Linux: CUDA Toolkit + NVIDIA Driver

---

### 7. Data Download Hangs

**Problem:** `fetch_data.py` hangs on Hugging Face download.

**Solution:**
```bash
# Set timeout
set HF_HUB_DOWNLOAD_TIMEOUT=600  # 10 minutes (Windows)

# Or download manually
huggingface-cli download zongowo111/v2-crypto-ohlcv-data --repo-type dataset
```

---

### 8. Notebook Kernel Crashes

**Error:** "The kernel died unexpectedly."

**Solutions:**

1. Restart kernel:
   - Jupyter: Kernel → Restart Kernel
   - Or press Ctrl+. twice

2. Clear notebook cache:
   ```bash
   jupyter kernelspec list
   jupyter kernelspec remove python3  # Remove problem kernel
   python -m ipykernel install --user  # Reinstall
   ```

3. Use simpler notebook:
   - Reduce data size
   - Use CSV instead of Parquet
   - Run on CPU only

---

### 9. Hugging Face API Quota Exceeded

**Error:** `HfHubHTTPError: 429 Client Error`

**Solution:**
```bash
# Add authentication
huggingface-cli login
# Enter your HF token from https://huggingface.co/settings/tokens

# Or in Python
from huggingface_hub import login
login(token="your_token_here")
```

---

### 10. Feature Engineering Takes Too Long

**Problem:** Feature calculation hangs for 30+ minutes.

**Optimization:**

1. Reduce lookback periods:
   ```yaml
   features:
     lookback_periods: [5, 10, 20]  # Default: [5, 10, 20, 50, 100, 200]
   ```

2. Use multiprocessing (in development):
   ```python
   # Manually parallel
   from multiprocessing import Pool
   ```

3. Use smaller dataset:
   ```python
   df = df.iloc[-50000:]  # Last 50k candles instead of all
   ```

---

## Model Performance Issues

### 11. Low Accuracy (Below 50%)

**Problem:** Model accuracy stuck around 20% (chance).

**Debugging:**

1. Check label distribution:
   ```python
   print(df['zigzag_label'].value_counts())
   # Should see all 5 classes (though imbalanced)
   ```

2. Verify ZigZag parameters:
   ```yaml
   zigzag:
     depth: 12       # Smaller = more signals
     deviation: 5    # Larger = fewer, clearer signals
   ```

3. Check feature quality:
   ```python
   print(df[feature_cols].describe())
   # Look for NaN, inf, or all-zero columns
   ```

4. Increase training time:
   ```bash
   python train.py --epochs 200 --batch_size 16
   ```

### 12. Model Overfitting (Train 90%, Test 50%)

**Problem:** Training accuracy high but test accuracy low.

**Solutions:**

1. Add regularization:
   ```python
   # In models.py
   model.add(Dropout(0.5))  # Increase from 0.3
   ```

2. Reduce model complexity:
   ```yaml
   model:
     lstm:
       layers: [64, 32]  # Smaller from [128, 64]
   ```

3. Use early stopping:
   ```python
   # Already in train.py
   early_stopping_patience=10  # Stop early
   ```

### 13. Class Imbalance Warning

**Warning:** "98% of labels are NO_SIGNAL"

**This is normal.** ZigZag signals are rare.

**To handle:**

1. Use weighted loss (recommended):
   ```python
   class_weight = {0: 1, 1: 100, 2: 100, 3: 100, 4: 100}
   model.fit(..., class_weight=class_weight)
   ```

2. Oversample minority classes:
   ```python
   from sklearn.utils import class_weight
   weights = class_weight.compute_class_weight(
       'balanced',
       classes=np.unique(y_train),
       y=y_train
   )
   ```

---

## Performance Optimization

### 14. Speed Up Training

1. Use smaller dataset:
   ```python
   df = df.sample(frac=0.5)  # Use 50% of data
   ```

2. Reduce features:
   ```yaml
   features:
     lookback_periods: [5, 10, 20]  # vs [5,10,20,50,100,200]
   ```

3. Smaller model:
   ```yaml
   model:
     lstm:
       layers: [64, 32]  # vs [128, 64]
   ```

4. Use GPU (if available):
   ```bash
   # Install CUDA first, TensorFlow auto-detects
   ```

---

### 15. Speed Up Inference

1. Quantize model (reduce size 10x):
   ```python
   import tensorflow as tf
   converter = tf.lite.TFLiteConverter.from_keras_model(model)
   tflite_model = converter.convert()
   ```

2. Batch predictions:
   ```python
   # Instead of one-by-one
   X_batch = np.array([X1, X2, X3, ...])  # (N, 60, 86)
   predictions = model.predict(X_batch)  # Much faster
   ```

3. Cache model in memory:
   ```python
   # Load once, reuse
   model = LSTMXGBoostModel()
   model.load('./models/trained')
   
   for data in stream:  # Don't reload each iteration
       pred = model.predict(data)
   ```

---

## Common Error Messages

| Error | Cause | Fix |
|-------|-------|-----|
| `ImportError: No module named...` | Missing package | `pip install -r requirements.txt` |
| `CUDA not found` | GPU drivers missing | Use CPU (OK) or install CUDA |
| `FileNotFoundError: model.h5` | Model not trained | Run `train.py` first |
| `KeyError: 'zigzag_label'` | Data not labeled | Run feature engineering first |
| `OOM when allocating` | Out of memory | Reduce batch_size |
| `OSError: Path too long` | Windows path issue | Use conda or shorter path |
| `urllib.error.URLError` | Network issue | Check internet, use proxy |
| `ParsingError in YAML` | config.yaml syntax | Check indentation, quotes |

---

## Getting Help

1. **Check logs:**
   ```bash
   # Save output to file
   python train.py > training.log 2>&1
   
   # Check for errors
   grep -i error training.log
   ```

2. **Enable debug mode:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

3. **File issue on GitHub:**
   - Include Python version: `python --version`
   - OS: Windows/Linux/Mac
   - Full error message
   - Steps to reproduce

4. **Search existing issues:**
   - https://github.com/caizongxun/zigzag-5class-prediction/issues

---

## Last Resort

If nothing works:

```bash
# Complete clean reinstall
deactivate
rmdir /s /q .venv
rmdir /s /q data\raw
rmdir /s /q models\trained

# Remove cached pip
pip cache purge

# Reinstall from scratch
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python tests/test_zigzag.py
```

If still failing, use **Google Colab** (free GPU, no setup needed):

```python
!git clone https://github.com/caizongxun/zigzag-5class-prediction
%cd zigzag-5class-prediction
!pip install -r requirements.txt
!python train.py
```

---

**Last Updated:** 2026-01-07
