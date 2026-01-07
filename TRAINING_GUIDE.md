# 完整訓練指南 - ZigZag 5-Class Prediction

## 概述

本指南涵蓋了完整的 10 步訓練管道：

1. **Step 1**: 下載分類器
2. **Step 2**: 獲取數據
3. **Step 3**: 應用 ZigZag 指標
4. **Step 4**: 特徵工程
5. **Step 5**: 時間序列分割
6. **Step 6**: 數據準備
7. **Step 7**: 創建 LSTM 序列
8. **Step 8**: 訓練模型
9. **Step 9**: 評估模型
10. **Step 10**: 保存模型

## 快速開始

### 基本訓練命令

```bash
# 使用預設設定訓練 BTCUSDT 15m
python train.py --symbol BTCUSDT --timeframe 15m

# 自定義參數
python train.py \
  --symbol BTCUSDT \
  --timeframe 15m \
  --epochs 50 \
  --batch_size 16 \
  --timesteps 60 \
  --zigzag_depth 12 \
  --zigzag_deviation 5

# 保存到特定目錄
python train.py \
  --symbol BTCUSDT \
  --timeframe 15m \
  --output ./my_models
```

## 詳細步驟說明

### Step 1: 下載分類器

```
Step 1: Downloading Symbol-Specific Classifier...
  Symbol: BTCUSDT
  Timeframe: 15m
  Note: Each symbol/timeframe has its OWN classification model
  Location: v1_model/BTCUSDT/15m/
```

**做了什麼：**
- 從 Hugging Face 數據集下載預訓練的分類器
- 每個 (symbol, timeframe) 組合都有獨立的模型
- 分類器位置: `v1_model/{SYMBOL}/{TIMEFRAME}/`

**常見問題：**
- 如果出現 `ERROR: Could not download classifier`
  - 檢查符號是否存在於 HF 數據集
  - 確認網絡連接
  - 驗證 HF Hub Token（如果需要）

### Step 2: 獲取數據

```
Step 2: Fetching Data...
Fetching BTCUSDT_15m data...
  Downloading: klines/BTCUSDT/BTC_15m.parquet...
BTC_15m.parquet: 100%|███████| 8.45M/8.45M [00:00<00:00, 32.1MB/s]

Data Summary:
  Shape: (219643, 7)
  Columns: ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
  Date range: 2019-09-23 08:30:00 to 2025-12-30 07:00:00
```

**做了什麼：**
- 從 HF 數據集下載 Parquet 格式的 OHLCV 數據
- 文件路徑: `klines/{SYMBOL}/{SYMBOL_PREFIX}_{TIMEFRAME}.parquet`
- 例如: `klines/BTCUSDT/BTC_15m.parquet`

**支援的幣種和時間框架：**

| 幣種 | 時間框架 |
|------|--------|
| BTCUSDT, ETHUSDT | 15m, 1h |
| BNBUSDT, XRPUSDT, SOLUSDT | 15m, 1h |
| ADAUSDT, LINKUSDT, MATICUSDT | 15m, 1h |
| 更多... | 15m, 1h |

### Step 3: 應用 ZigZag 指標

```
Step 3: Applying ZigZag Indicator...
  Depth: 12, Deviation: 5%
  ZigZag Labels:
    NO_SIGNAL (0): 174234 (79.3%)
    HH (1): 11245 (5.1%)
    LH (2): 8956 (4.1%)
    HL (3): 13123 (6.0%)
    LL (4): 12085 (5.5%)
```

**做了什麼：**
- 計算 ZigZag 樞軸點
- 標籤化每根 K 線的價格結構
- 5 個類別:
  - `0 (NO_SIGNAL)`: 無明確信號
  - `1 (HH)`: 高位到更高位 (看漲繼續)
  - `2 (LH)`: 低位到高位 (看漲轉折)
  - `3 (HL)`: 高位到低位 (看跌轉折)
  - `4 (LL)`: 低位到更低位 (看跌繼續)

**參數調整：**
```bash
# 更敏感的 ZigZag (捕捉較小的反轉)
python train.py --zigzag_depth 8 --zigzag_deviation 3

# 不敏感的 ZigZag (捕捉主要趨勢)
python train.py --zigzag_depth 20 --zigzag_deviation 7
```

### Step 4: 特徵工程

```
Step 4: Feature Engineering...
  Total features created: 86
  Sample features: ['high_low_range', 'open_close_range', ...
  NaN values removed
```

**創建的特徵類別：**

1. **基本特徵** (8)
   - `high_low_range`, `open_close_range`, `true_range`
   - 價格變化: `price_change_1`, `price_pct_change_5`, ...

2. **移動平均** (15)
   - SMA: `sma_5`, `sma_10`, `sma_20`, `sma_50`, `sma_200`
   - EMA: `ema_5`, `ema_10`, `ema_20`, ...
   - 比率: `price_ma_ratio_20`, `trend_strength_20`

3. **動量指標** (12)
   - RSI: `rsi_5`, `rsi_10`, `rsi_14`, `rsi_20`
   - MACD: `macd_line_12`, `macd_line_26`
   - Momentum: `momentum_5`, `momentum_10`, `momentum_20`

4. **波動率** (9)
   - ATR: `atr_5`, `atr_10`, `atr_20`, `atr_pct_20`
   - 波動率: `volatility_5`, `volatility_20`, ...
   - 比率: `volatility_ratio_20_5`

5. **成交量** (6)
   - `volume_ma_5`, `volume_ma_20`
   - `volume_ratio_5`, `volume_ratio_20`
   - `obv` (On Balance Volume)

6. **結構特徵** (8)
   - 高/低點: `highest_high_20`, `lowest_low_50`
   - 距離: `distance_resistance_20`, `distance_support_50`
   - 連續柱: `consecutive_bars`

7. **進階指標** (8)
   - 布林帶: `bb_upper_20`, `bb_lower_20`, `bb_width_20`, `bb_position_20`
   - 其他: `volatility_regime`, `price_position_20`

### Step 5: 時間序列分割

```
Step 5: Time Series Split...
  Total samples: 219643
  Train: 153750 (70.0%)
  Validation: 32946 (15.0%)
  Test: 32947 (15.0%)
```

**重要：使用時間序列分割而不是隨機分割**

```python
# ✅ 正確做法：時間序列分割
train_df = df.iloc[:int(0.7*len(df))]       # 2019-2023
val_df = df.iloc[int(0.7*len(df)):int(0.85*len(df))]    # 2023-2024
test_df = df.iloc[int(0.85*len(df)):]       # 2024-2025

# ❌ 錯誤做法：隨機分割（數據洩漏）
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df)  # 不要這樣做！
```

### Step 6: 數據準備

```
Step 6: Data Preparation...
  Features normalized using StandardScaler
  X_train shape: (153750, 86)
  X_val shape: (32946, 86)
  X_test shape: (32947, 86)
  Feature mean: -0.000000, std: 1.000000
```

**做了什麼：**
- 提取特徵和標籤
- 使用 StandardScaler 進行標準化 (基於訓練集)
- 驗證數據範圍

### Step 7: 創建 LSTM 序列

```
Step 7: Creating Sequences for LSTM...
  Sequence length (timesteps): 60
  Train sequences: (153690, 60, 86)
  Val sequences: (32886, 60, 86)
  Test sequences: (32887, 60, 86)
  Class distribution in train: (array([0, 1, 2, 3, 4]), ...)
```

**時間窗口概念：**

```
Original sequence:  [t-59, t-58, ..., t-1, t, t+1, ...]
                    ↓
LSTM 序列:          [t-59:t] → label[t+1]
                    [t-58:t+1] → label[t+2]
                    ...
```

**參數調整：**
```bash
# 更短的上下文（快速變動市場）
python train.py --timesteps 30

# 更長的上下文（趨勢交易）
python train.py --timesteps 120
```

### Step 8: 訓練模型

```
Step 8: Training LSTM Model...
  Epochs: 50
  Batch size: 16
  Early stopping patience: 10

  Training Results:
    LSTM Train Accuracy: 0.6234
    LSTM Val Accuracy: 0.5987
```

**模型架構：**

```
Input (60, 86)
  ↓
LSTM(128) + Dropout(0.3)
  ↓
LSTM(64) + Dropout(0.3)
  ↓
Dense(32) + ReLU + Dropout(0.2)
  ↓
Dense(16) + ReLU
  ↓
Dense(5) + Softmax
  ↓
Output (5-class probabilities)
```

**訓練技巧：**
- Early Stopping: 如果驗證損失不改善超過 10 個 epoch，停止訓練
- Dropout: 防止過擬合
- StandardScaler: 確保特徵在相同範圍內

### Step 9: 評估模型

```
Step 9: Evaluating Model on Test Set...

  Test Set Performance:
    Accuracy: 0.6234
    Precision: 0.6145
    Recall: 0.6234
    F1-Score: 0.6178

  Confusion Matrix:
         NO_SIGNAL          HH          LH          HL          LL
NO_S        24532        1234        1098        1956        2127
HH           1298        1856         234         345         267
LH           1123         289        1567         234         673
HL           1834         456         234        1945         654
LL           2134         567         678         645        1183
```

**指標解釋：**
- **Accuracy**: 正確預測的百分比
- **Precision**: 預測為正類的樣本中，有多少實際是正類
- **Recall**: 實際為正類的樣本中，有多少被正確預測
- **F1-Score**: Precision 和 Recall 的調和平均值

### Step 10: 保存模型

```
Step 10: Saving Models...
  Model saved to models/trained/BTCUSDT/15m
  Scaler saved to models/trained/BTCUSDT/15m/scaler.pkl
  Feature columns saved to models/trained/BTCUSDT/15m/feature_columns.pkl
  Metadata saved to models/trained/BTCUSDT/15m/metadata.json
```

**保存的文件：**
```
models/trained/BTCUSDT/15m/
├── lstm_model.h5          # 訓練的 LSTM 模型
├── config.json            # 模型配置 (timesteps, n_features, n_classes)
├── scaler.pkl             # 特徵標準化器
├── feature_columns.pkl    # 特徵列名
└── metadata.json          # 訓練元數據 (準確率、F1-Score 等)
```

## 推理/預測

訓練完成後，使用 `infer.py` 進行預測：

```bash
python infer.py --symbol BTCUSDT --timeframe 15m
```

詳見 [infer.py](./infer.py) 文檔。

## 性能優化

### GPU 加速

```python
# 自動使用 GPU（如果可用）
import tensorflow as tf
print("GPU 可用:", tf.config.list_physical_devices('GPU'))
```

### 批量訓練多個符號

```bash
#!/bin/bash
for symbol in BTCUSDT ETHUSDT BNBUSDT; do
  for timeframe in 15m 1h; do
    python train.py --symbol $symbol --timeframe $timeframe
  done
done
```

## 故障排查

### OOM (內存不足)

```bash
# 減少批大小
python train.py --batch_size 8

# 減少時間步
python train.py --timesteps 30

# 減少特徵（編輯 FeatureEngineer）
```

### 低準確率

```bash
# 增加訓練輪次
python train.py --epochs 200

# 調整 ZigZag 參數
python train.py --zigzag_depth 10 --zigzag_deviation 4

# 增加時間步
python train.py --timesteps 90
```

### 模型保存失敗

```bash
# 檢查輸出目錄
ls -la models/trained/BTCUSDT/15m/

# 確保有寫入權限
chmod -R 755 models/
```

## 常見參數組合

### 高頻交易 (15m)
```bash
python train.py \
  --symbol BTCUSDT \
  --timeframe 15m \
  --timesteps 40 \
  --zigzag_depth 8 \
  --zigzag_deviation 3 \
  --epochs 100 \
  --batch_size 32
```

### 中期趨勢 (1h)
```bash
python train.py \
  --symbol BTCUSDT \
  --timeframe 1h \
  --timesteps 60 \
  --zigzag_depth 12 \
  --zigzag_deviation 5 \
  --epochs 50 \
  --batch_size 16
```

## 下一步

1. ✅ 完成訓練
2. → 使用 `infer.py` 進行預測
3. → 部署到生產環境
4. → 設置監控和重新訓練流程

## 文檔

- [README.md](./README.md) - 項目概述
- [QUICK_START.md](./QUICK_START.md) - 快速開始
- [ARCHITECTURE.md](./ARCHITECTURE.md) - 系統架構
- [infer.py](./infer.py) - 推理文檔

---

**最後更新**: 2026-01-07
**版本**: 1.0
