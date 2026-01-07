# Deployment Guide

## Local Development

### Prerequisites
- Python 3.9+
- 4GB+ RAM
- GPU (optional, NVIDIA with CUDA)

### Step 1: Environment Setup

```bash
# Clone
git clone https://github.com/caizongxun/zigzag-5class-prediction
cd zigzag-5class-prediction

# Virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# Dependencies
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
python tests/test_zigzag.py
# Expected: All tests passed!
```

### Step 3: Download Data

```bash
python data/fetch_data.py
# Downloads BTC 15m + Stage 1 classifier
```

### Step 4: Train Model

**Option A: Jupyter Notebooks (Interactive)**

```bash
jupyter notebook
```

Run notebooks in order:
1. 01_data_exploration.ipynb
2. 02_feature_engineering.ipynb
3. 03_model_training.ipynb

**Option B: Command Line (Batch)**

```bash
python train.py --config config.yaml --output ./models/trained
```

### Step 5: Inference

```bash
python infer.py --model ./models/trained --symbol BTCUSDT --timeframe 15m
```

## Production Deployment

### Option 1: Flask API Service

```python
# app.py
from flask import Flask, jsonify
from src.models import LSTMXGBoostModel

app = Flask(__name__)
model = LSTMXGBoostModel()
model.load('./models/trained')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    X = np.array(data['features']).reshape(1, 60, -1)
    signal, confidence = model.predict(X)
    return jsonify({'signal': signal, 'confidence': float(confidence)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

Run:
```bash
pip install flask
python app.py
```

Test:
```bash
curl -X POST http://localhost:5000/predict -d @data.json
```

### Option 2: Docker Containerization

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t zigzag-predictor .
docker run -p 5000:5000 zigzag-predictor
```

### Option 3: AWS Lambda

1. Package model:
```bash
zip -r lambda_package.zip . -x "*.git*" "data/raw/*"
```

2. Create Lambda function (Python 3.9)

3. Upload code and set handler to `lambda_handler`

4. Test with sample event

### Option 4: Kubernetes

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: zigzag-predictor
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: predictor
        image: zigzag-predictor:latest
        ports:
        - containerPort: 5000
```

Deploy:
```bash
kubectl apply -f deployment.yaml
```

## Monitoring

### Logging

```python
import logging
logging.basicConfig(filename='predictions.log', level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f'Prediction: {signal}, Confidence: {confidence}')
```

### Metrics

Track:
- Prediction accuracy on holdout test set
- Inference latency
- Model drift (retrain if accuracy drops)
- Class distribution over time

### Retraining Schedule

```bash
# Weekly retraining
0 2 * * 0 /path/to/retrain.py
```

## Performance Tuning

### Optimize Inference

1. Batch predictions:
```python
# Process 100 samples at once
X_batch = np.array([...])  # (100, 60, 86)
predictions, confidences = model.predict(X_batch)
```

2. Model quantization (reduce size 10x):
```python
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model(model)
quantized_model = converter.convert()
```

3. Use GPU inference:
```python
with tf.device('/GPU:0'):
    predictions = model.predict(X)
```

## Security

### Protect API

```python
from functools import wraps
from flask import request, abort

API_KEY = 'your-secret-key'

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if request.headers.get('X-API-Key') != API_KEY:
            abort(401)
        return f(*args, **kwargs)
    return decorated

@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    # ...
```

### Model versioning

```bash
# Save with version
model.save('./models/trained/v1.0')
model.save('./models/trained/v1.1')

# In config: model_version = 'v1.1'
```

## Disaster Recovery

### Backup Strategy

```bash
# Daily backups
0 3 * * * tar -czf models_backup_$(date +%Y%m%d).tar.gz models/
```

### Fallback Logic

```python
try:
    model = LSTMXGBoostModel()
    model.load('./models/current')
except Exception as e:
    logger.error(f'Failed to load current model: {e}')
    logger.info('Loading backup model...')
    model.load('./models/backup')
```

## Cost Optimization

- Use spot instances for training
- Batch inference during off-peak hours
- Compress model (quantization reduces size 10x)
- Cache predictions for repeated inputs
- Use AutoML for hyperparameter tuning

## Compliance

- Audit prediction logs
- Document model decisions
- Regular accuracy audits
- Data retention policies
- GDPR compliance for data

## Checklist

Before production:

- [ ] Model accuracy >= 70%
- [ ] Inference latency < 100ms
- [ ] API authentication enabled
- [ ] Logging and monitoring set up
- [ ] Backup and recovery tested
- [ ] Load testing completed (1000+ req/s)
- [ ] Documentation updated
- [ ] Security audit passed

