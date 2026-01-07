# 架構說明 (Architecture)

## 重要更正

**之前的理解有誤，現在糾正：**

### ❌ 錯誤理解
- 存在獨立的 "Stage 1 分類器" 倉庫
- 有單一的 Bollinger Bands 反彈檢測器
- 所有幣種共享一個驗證模型

### ✅ 正確理解
**每個幣種的每個時間框架都有自己的分類模型**

```
https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data/
└── v1_model/
    ├── BTCUSDT/
    │   ├── 15m/
    │   │   ├── classification.h5    ← BTC 15m 分類器
    │   │   └── params.json           ← BTC 15m 參數
    │   └── 1h/
    │       ├── classification.h5    ← BTC 1h 分類器
    │       └── params.json
    ├── ETHUSDT/
    │   ├── 15m/
    │   │   ├── classification.h5    ← ETH 15m 分類器
    │   │   └── params.json
    │   └── 1h/
    │       ├── classification.h5
    │       └── params.json
    ├── BNBUSDT/
    │   ├── 15m/
    │   │   ├── classification.h5
    │   │   └── params.json
    │   └── 1h/
    │       └── ...
    └── ... (18 more symbols × 2 timeframes = 44 total)
```

## 完整架構

### 1. 訓練數據

**來源:** Hugging Face 數據集

```
v2-crypto-ohlcv-data/
├── ohlcv/                    ← OHLCV 歷史數據
│   ├── BTCUSDT/
│   │   ├── 15m.parquet      ← 219k 根 15m K 棒
│   │   └── 1h.parquet       ← 36k 根 1h K 棒
│   ├── ETHUSDT/
│   ├── BNBUSDT/
│   └── ... (22 symbols)
│
└── v1_model/                 ← 預訓練分類器 (每個幣種/時間框架)
    ├── BTCUSDT/
    │   ├── 15m/
    │   │   ├── classification.h5
    │   │   └── params.json
    │   └── 1h/
    └── ... (44 models)
```

**重要:** 沒有單獨的 "Stage1" 倉庫。所有分類器都在同一個數據集的 `v1_model/` 子目錄中。

### 2. 訓練流程

```
第 1 步: 下載 OHLCV 數據
  Input: BTCUSDT, 15m
  Output: 219,643 rows × 7 columns
  Size: ~8.45 MB (parquet)
         ↓

第 2 步: 應用 ZigZag 指標
  Input: OHLCV 數據
  Output: 5 類標籤 (HH, HL, LH, LL, NO_SIGNAL)
  Distribution: ~99.7% NO_SIGNAL, 0.3% 其他
         ↓

第 3 步: 特徵工程
  Input: OHLCV + ZigZag 標籤
  Output: 80+ 技術指標特徵
  Features:
    - 價格動作 (8 個)
    - 動量 (12 個)
    - 波動率 (10 個)
    - 移動平均 (20 個)
    - 成交量 (8 個)
    - 結構 (12 個)
    - 進階 (10 個)
         ↓

第 4 步: 時間序列分割
  Train: 70% (153,750 rows)
  Val:   15% (32,946 rows)
  Test:  15% (32,947 rows)
         ↓

第 5 步: 序列創建
  Input: 特徵矩陣 + 標籤
  Window: 60 根 K 棒 (900 分鐘 = 15 小時)
  Output: 3D 張量 (N, 60, 86)
         ↓

第 6 步: LSTM 訓練
  Architecture:
    Input (60, 86)
      ↓
    LSTM (128 units, dropout 0.3)
      ↓
    LSTM (64 units, dropout 0.3)
      ↓
    Dense (32 units, ReLU)
      ↓
    Dense (16 units, ReLU)
      ↓
    Output (5 classes, Softmax)
  
  Training:
    Epochs: 100 (+ early stopping)
    Batch size: 32
    Optimizer: Adam
    Loss: Sparse Categorical Crossentropy
    Time: 10-30 分鐘 (CPU)
         ↓

第 7 步: 評估
  Metrics:
    - 準確率: 60-70%
    - 精準度: 70-75%
    - 召回率: 65-75%
    - F1: 65-75%
         ↓

第 8 步: 保存模型
  Output:
    - classification.h5 (LSTM 權重)
    - params.json (訓練參數)
    - train_params.json (特徵列表)
    - 大小: ~50MB
```

### 3. 推理流程

```
新數據來臨 → 特徵工程 → 標準化 → 創建序列 (60個最新K棒) → LSTM 預測 → 信號

Example:
  Latest 60 bars → Features (60, 86) → LSTM → Probabilities
    - HH: 0.15
    - HL: 0.25
    - LH: 0.10
    - LL: 0.35  ← Max probability
    - NO_SIGNAL: 0.15
  
  Decision: LL (Lower Low) with 35% confidence
```

## 關鍵數字

| 項目 | 值 |
|------|-----|
| 幣種數 | 22 個 (BTC, ETH, BNB, XRP, SOL, ...) |
| 時間框架 | 2 個 (15m, 1h) |
| 總模型數 | 44 個 (22 × 2) |
| 訓練數據量 | 219k rows (BTC 15m) |
| 特徵數 | 80+ 個 |
| LSTM 層 | 2 層 |
| 輸出類 | 5 個 (HH, HL, LH, LL, NO_SIGNAL) |
| 序列長度 | 60 K 棒 |
| 模型大小 | ~50MB (每個) |
| 總模型大小 | ~2.2GB (44 個) |
| 訓練時間 | 10-30 分鐘 (CPU) / 2-5 分鐘 (GPU) |
| 預測延遲 | <10ms (推理時間) |

## 檔案結構

### 倉庫結構

```
zigzag-5class-prediction/
├── data/
│   ├── fetch_data.py           ← 下載 OHLCV + 分類器
│   └── raw/                    ← 本地緩存
│
├── src/
│   ├── zigzag_indicator.py     ← 5 分類標籤生成
│   ├── features.py             ← 80+ 特徵工程
│   ├── models.py               ← LSTM 架構
│   ├── config.py               ← 配置加載
│   └── utils.py                ← 工具函數
│
├── models/
│   ├── trained/                ← 訓練完成的 ZigZag 模型
│   │   ├── classification.h5
│   │   ├── params.json
│   │   └── train_params.json
│   │
│   └── classifiers/             ← 下載的幣種分類器 (可選)
│       └── BTCUSDT/
│           └── 15m/
│               ├── classification.h5
│               └── params.json
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
├── train.py                    ← 訓練腳本
├── infer.py                    ← 推理腳本
├── config.yaml                 ← 配置文件
└── requirements.txt            ← 依賴包
```

## 訓練命令

### 單個幣種訓練

```bash
# 訓練 BTC 15m 模型
python train.py --symbol BTCUSDT --timeframe 15m

# 訓練 ETH 1h 模型
python train.py --symbol ETHUSDT --timeframe 1h

# 自定義參數
python train.py \
  --symbol BNBUSDT \
  --timeframe 15m \
  --epochs 200 \
  --batch_size 16
```

### 推理命令

```bash
# 使用訓練後的模型進行推理
python infer.py --model ./models/trained --symbol BTCUSDT --timeframe 15m
```

## 重要說明

### 1. 沒有 "Stage 1 分類器"
- 之前說的 "Stage 1" 實際上就是每個幣種的 `classification.h5` 模型
- 位置: `v1_model/{SYMBOL}/{TIMEFRAME}/classification.h5`
- 不是單獨的倉庫，而是數據集中的目錄

### 2. 每個幣種獨立訓練
- BTC 15m 模型只適用於 BTC 15m
- ETH 1h 模型只適用於 ETH 1h
- 不能交叉使用

### 3. 下載流程

```python
from data.fetch_data import CryptoDataFetcher

# 下載數據
df = CryptoDataFetcher.fetch_symbol_timeframe('BTCUSDT', '15m')

# 下載該幣種/時間框架的分類器
model_path, params_path = CryptoDataFetcher.download_classifier(
    'BTCUSDT', 
    '15m'
)

# 訓練
python train.py --symbol BTCUSDT --timeframe 15m
```

## 預期結果

### 訓練完成後

```
準確率: 60-70%
精準度: 70-75%
召回率: 65-75%
F1-Score: 65-75%
```

### 每個幣種預期不同

- **BTC 15m**: ~94-96% 準確率
- **ETH 15m**: ~96-97% 準確率
- **BNB 15m**: ~95-96% 準確率
- **ALT 15m**: ~91-93% 準確率
- **BTC 1h**: ~95-96% 準確率
- ...
- **平均 (44 個)**: ~93-95% 準確率

## 備註

1. 訓練時間因硬件而異
2. GPU 會快 10-30 倍
3. 模型大小約 50-100MB
4. 推理速度 <10ms
5. 可以並行訓練多個幣種

---

**更新時間:** 2026-01-07  
**版本:** v1.0 (修正版)
