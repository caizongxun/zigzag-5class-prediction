# 快速開始 - 5 分鐘上手

## 前置要求

- Python 3.8+
- 虛擬環境已激活
- 依賴已安裝 (`pip install -r requirements.txt`)

## 第一次使用 (10 分鐘)

### 步驟 1: 訓練模型 (5-10 分鐘)

```bash
# 使用改進的參數訓練
python train.py \
  --symbol BTCUSDT \
  --timeframe 15m \
  --zigzag_depth 9 \
  --zigzag_deviation 2.5 \
  --epochs 50 \
  --batch_size 32
```

**預期輸出:**
```
Step 1: Downloading Symbol-Specific Classifier...
✓ BTCUSDT_15m classifier downloaded successfully

Step 2: Fetching Data...
Fetching BTCUSDT_15m data...
100%|████████████| 8.45M/8.45M

Step 3: Applying ZigZag Indicator...
ZigZag Labels:
  NO_SIGNAL (0): 170000 (77.4%)
  HH (1): 11000 (5.0%)
  LH (2): 11000 (5.0%)
  HL (3): 13000 (6.0%)
  LL (4): 14000 (6.4%)

... (特徵工程、訓練等)

TRAINING COMPLETE
======================================================================
Model Information:
  Symbol: BTCUSDT
  Timeframe: 15m
  Total samples: 219643

Model Performance:
  Test Accuracy: 0.5823
  Test F1-Score: 0.5412
...
```

✅ **訓練完成！** 模型已保存到 `models/trained/BTCUSDT/15m/`

### 步驟 2: 進行推理 (1 分鐘)

```bash
python infer.py --symbol BTCUSDT --timeframe 15m
```

**預期輸出:**
```
Step 1: Loading Model Configuration...
  Features: 80
  Timesteps: 60
  Classes: 5

Step 2: Loading LSTM Model...
  ✓ Model loaded successfully

Step 3: Fetching Latest Data...
  Data shape: (219643, 7)

Step 4: Applying ZigZag Indicator...
  ✓ ZigZag labels applied

Step 5: Feature Engineering...
  ✓ Features computed

Step 6: Making Predictions...

============================================================
LATEST PREDICTION
============================================================
  Signal: HH (ID: 1)
  Confidence: 67.45%
  Timestamp: 2025-12-30 07:00:00
  Price: $43567.89
  High: $43598.00
  Low: $43445.00
  Volume: 1234567
============================================================

LAST 10 CANDLES
...
```

✅ **預測完成！** 看到 `HH` 信號表示看漲繼續。

---

## 使用不同的符號

### 訓練其他幣種

```bash
# 訓練 ETHUSDT
python train.py --symbol ETHUSDT --timeframe 15m --epochs 50

# 訓練 BNBUSDT
python train.py --symbol BNBUSDT --timeframe 15m --epochs 50

# 訓練 1h 時間框架
python train.py --symbol BTCUSDT --timeframe 1h --epochs 50
```

### 推理其他幣種

```bash
# 推理 ETHUSDT
python infer.py --symbol ETHUSDT --timeframe 15m

# 推理 BNBUSDT
python infer.py --symbol BNBUSDT --timeframe 15m
```

---

## 訓練參數說明

### 快速訓練 (2-3 分鐘)
```bash
python train.py \
  --symbol BTCUSDT \
  --timeframe 15m \
  --epochs 20 \
  --batch_size 64
```

### 標準訓練 (5-10 分鐘)
```bash
python train.py \
  --symbol BTCUSDT \
  --timeframe 15m \
  --epochs 50 \
  --batch_size 32
```

### 深度訓練 (15-20 分鐘)
```bash
python train.py \
  --symbol BTCUSDT \
  --timeframe 15m \
  --epochs 100 \
  --batch_size 16
```

### 高頻交易 (15m 高敏感)
```bash
python train.py \
  --symbol BTCUSDT \
  --timeframe 15m \
  --zigzag_depth 6 \
  --zigzag_deviation 1.5 \
  --epochs 100 \
  --batch_size 16
```

### 中期趨勢 (1h 平衡)
```bash
python train.py \
  --symbol BTCUSDT \
  --timeframe 1h \
  --zigzag_depth 10 \
  --zigzag_deviation 2.5 \
  --epochs 50 \
  --batch_size 32
```

---

## 信號解釋

| 信號 | 含義 | 交易建議 |
|------|------|--------|
| **NO_SIGNAL (0)** | 沒有明確結構 | 觀望 |
| **HH (1)** | 高位到更高位 | 看漲延續信號 ✅ |
| **LH (2)** | 低位到高位 | 看漲反轉信號 ✅ |
| **HL (3)** | 高位到低位 | 看跌反轉信號 ⚠️ |
| **LL (4)** | 低位到更低位 | 看跌延續信號 ❌ |

**置信度 (Confidence):**
- > 70%: 強信號
- 50-70%: 中等信號
- 30-50%: 弱信號
- < 30%: 不確定

---

## 常見問題

### Q: 訓練需要多長時間?
A: 使用 CPU 約 5-20 分鐘，取決於硬件和 epoch 數。使用 GPU 會快 5-10 倍。

### Q: 模型文件在哪裡?
A: `models/trained/{SYMBOL}/{TIMEFRAME}/`

### Q: 如何重新訓練?
A: 
```bash
rm -r models/trained/BTCUSDT/15m
python train.py --symbol BTCUSDT --timeframe 15m
```

### Q: 推理出現錯誤?
A: 確保模型已訓練:
```bash
# 檢查模型文件
ls -la models/trained/BTCUSDT/15m/

# 重新訓練
python train.py --symbol BTCUSDT --timeframe 15m
```

### Q: 如何調整靈敏度?
A: 調整 ZigZag 參數:
```bash
# 更靈敏 (捕捉小反轉)
python train.py --zigzag_depth 6 --zigzag_deviation 1.5

# 更保守 (只捕捉大趨勢)
python train.py --zigzag_depth 12 --zigzag_deviation 4.0
```

---

## 下一步

- 📖 詳細訓練指南: [TRAINING_GUIDE.md](./TRAINING_GUIDE.md)
- 🐛 故障排除: [OVERFITTING_AND_FIXES.md](./OVERFITTING_AND_FIXES.md)
- 🏗️ 系統架構: [ARCHITECTURE.md](./ARCHITECTURE.md)
- 📚 項目概述: [README.md](./README.md)

---

## 支持的幣種和時間框架

**支持的幣種 (22 個):**
- BTC (BTCUSDT)
- ETH (ETHUSDT)
- BNB, XRP, SOL, ADA
- LINK, MATIC, FIL, LTC
- DOGE, SHIB, OP, ARB
- APE, BLUR, LDO, STX
- UNI, AAVE, GRT, SAND

**支持的時間框架:**
- 15m (15 分鐘)
- 1h (1 小時)

---

**最後更新**: 2026-01-07
**版本**: 1.0
**狀態**: ✅ 就緒
