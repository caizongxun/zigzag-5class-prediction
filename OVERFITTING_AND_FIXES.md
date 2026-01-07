# 過擬合問題診斷與修復

## 問題診斷

### 症狀 1: 準確率 100% (過於完美)

當你看到這樣的輸出:
```
Test Accuracy: 1.0000
F1-Score: 1.0000
```

**這通常表示過擬合，而不是模型真的完美！**

### 症狀 2: 類別不平衡

```
ZigZag Labels:
  NO_SIGNAL (0): 219616 (100.0%)
  LH (2): 27 (0.0%)
  HH, HL, LL: 0
```

**99.988% 的數據是一個類別，模型簡單地學會了預測多數類！**

### 症狀 3: Loss 變成 NaN

```
Epoch 5/50
loss: nan - accuracy: 0.9998 - val_loss: nan
```

**學習率太高或梯度爆炸**

---

## 根本原因分析

### 原因 1: ZigZag 參數過於嚴格

**舊設置:**
```bash
--zigzag_depth 12 --zigzag_deviation 5%
```

**結果:** 幾乎所有樣本都被標籤為 NO_SIGNAL

**解釋:**
- `depth=12`: 只在最後 12 根 K 線中尋找極值點 → 高度過濾
- `deviation=5%`: 需要 5% 的價格變化才能確認反轉 → 太嚴格

### 原因 2: 類別失衡導致的虛假準確率

模型只需要預測「NO_SIGNAL」就能達到 99.988% 準確率：

```python
# 這是模型實際做的
def model_strategy():
    return predict_NO_SIGNAL_always()  # 99.988% 準確
```

混淆矩陣顯示:
```
NO_SIGNAL: 32887 (all correct)
Other classes: 0 (not predicted at all)
```

---

## 解決方案

### ✅ 解決方案 1: 調整 ZigZag 參數

**改進的設置:**

#### 高頻交易 (15m)
```bash
python train.py \
  --symbol BTCUSDT \
  --timeframe 15m \
  --zigzag_depth 8 \
  --zigzag_deviation 2.0 \
  --epochs 100 \
  --batch_size 16
```

**結果預期:**
```
ZigZag Labels:
  NO_SIGNAL (0): 170000 (77%)
  HH (1): 12000 (5%)
  LH (2): 11000 (5%)
  HL (3): 13000 (6%)
  LL (4): 13000 (6%)
```

#### 中期趨勢 (1h)
```bash
python train.py \
  --symbol BTCUSDT \
  --timeframe 1h \
  --zigzag_depth 10 \
  --zigzag_deviation 3.0 \
  --epochs 50 \
  --batch_size 32
```

#### 激進的反轉捕捉
```bash
python train.py \
  --symbol BTCUSDT \
  --timeframe 15m \
  --zigzag_depth 6 \
  --zigzag_deviation 1.5 \
  --epochs 100 \
  --batch_size 8
```

### ✅ 解決方案 2: 使用加權損失函數

修改 `src/models.py` 的損失函數:

```python
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# 計算類權重
class_weight = {
    0: 1.0,     # NO_SIGNAL
    1: 10.0,    # HH (少見類別)
    2: 10.0,    # LH
    3: 9.0,     # HL
    4: 9.0      # LL
}

# 在 train() 方法中使用
history = self.lstm_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    class_weight=class_weight,  # 加入類權重
    callbacks=[early_stop],
    verbose=1
)
```

### ✅ 解決方案 3: 正確的評估指標

不要只看準確率，要看：

```python
# 對於不平衡數據，F1-Score 更重要
from sklearn.metrics import balanced_accuracy_score

# 計算平衡準確率
balanced_acc = balanced_accuracy_score(y_test, predicted_classes)
print(f"Balanced Accuracy: {balanced_acc:.4f}")  # 對所有類別平等對待
```

### ✅ 解決方案 4: 分層抽樣

```python
from sklearn.model_selection import StratifiedKFold

# 使用分層 K 折交叉驗證
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # 每折都保持類別分佈
```

---

## 推薦的訓練參數

### 場景 1: 平衡準確性和信號捕捉

```bash
python train.py \
  --symbol BTCUSDT \
  --timeframe 15m \
  --zigzag_depth 9 \
  --zigzag_deviation 2.5 \
  --epochs 100 \
  --batch_size 32 \
  --timesteps 60
```

**預期結果:**
- 準確率: 55-65%
- F1-Score: 52-60%
- 信號捕捉: 良好（能檢測反轉）

### 場景 2: 更好的少見類別捕捉

```bash
python train.py \
  --symbol BTCUSDT \
  --timeframe 15m \
  --zigzag_depth 6 \
  --zigzag_deviation 1.5 \
  --epochs 150 \
  --batch_size 16 \
  --timesteps 90
```

**預期結果:**
- 準確率: 50-58%
- F1-Score: 48-56%
- 反轉捕捉: 非常好
- 假信號: 可能增加

### 場景 3: 保守的趨勢跟踪

```bash
python train.py \
  --symbol BTCUSDT \
  --timeframe 1h \
  --zigzag_depth 12 \
  --zigzag_deviation 4.0 \
  --epochs 50 \
  --batch_size 32 \
  --timesteps 120
```

**預期結果:**
- 準確率: 60-68%
- F1-Score: 58-65%
- 假信號: 最少
- 漏失信號: 可能更多

---

## 快速測試

### 步驟 1: 測試不同的 ZigZag 參數

```bash
# 參數 A (敏感)
python train.py --symbol BTCUSDT --timeframe 15m --zigzag_depth 6 --zigzag_deviation 1.5

# 參數 B (平衡)
python train.py --symbol BTCUSDT --timeframe 15m --zigzag_depth 9 --zigzag_deviation 2.5

# 參數 C (保守)
python train.py --symbol BTCUSDT --timeframe 15m --zigzag_depth 12 --zigzag_deviation 5.0
```

### 步驟 2: 比較標籤分佈

查看訓練輸出中的 `ZigZag Labels` 部分，選擇分佈最均衡的參數組合。

### 步驟 3: 檢查評估指標

查看完整的混淆矩陣和每個類別的 F1-Score，而不是整體準確率。

---

## 如何檢查過擬合

### 紅旗清單

- [ ] 訓練準確率 > 95% 且驗證準確率 > 90%
- [ ] 所有類別的混淆矩陣都在對角線上
- [ ] Loss 函數變成 NaN
- [ ] 某個類別的比例 > 99%
- [ ] 模型預測只有一個類別
- [ ] F1-Score 接近 1.0

### 綠旗清單 (健康的模型)

- [ ] 訓練準確率 60-70%，驗證準確率 58-68%
- [ ] 訓練 Loss 和驗證 Loss 一起下降
- [ ] 類別分佈相對均衡（沒有 > 80% 單一類別）
- [ ] F1-Score 在 0.55-0.70 之間
- [ ] 混淆矩陣顯示所有類別都有預測

---

## 調試檢查清單

### 訓練前

```python
# 1. 檢查標籤分佈
label_dist = df['zigzag_label'].value_counts()
print(label_dist / len(df))  # 應該沒有 > 90% 的類別

# 2. 檢查特徵統計
print(X_train.mean(axis=0))  # 應該接近 0（已標準化）
print(X_train.std(axis=0))   # 應該接近 1（已標準化）

# 3. 檢查序列長度
print(f"Sequences: {X_train_seq.shape}")  # (samples, 60, 86)
```

### 訓練期間

```
每個 epoch 檢查:
- Loss 是否穩定下降（不是震盪或上升）
- Val Loss 是否跟隨 Train Loss
- Accuracy 是否合理 (50-70%，不是 > 99%)
- 是否出現 NaN
```

### 訓練後

```bash
# 檢查混淆矩陣
python -c "import numpy as np; from sklearn.metrics import confusion_matrix; cm = confusion_matrix([0]*100 + [1]*50 + [2]*30); print(cm)"

# 每個類別的精度
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
```

---

## 常見的 ZigZag 參數組合

| 用途 | Depth | Deviation | 預期標籤分佈 | 適用場景 |
|-----|-------|-----------|------------|--------|
| 極端敏感 | 5-6 | 1.0-1.5% | 70-15-5-5-5 | 極短期交易 |
| 敏感 | 7-8 | 1.5-2.0% | 75-10-8-4-3 | 短期交易 |
| **推薦** | **9-10** | **2.0-2.5%** | **77-7-6-5-5** | 一般用途 |
| 平衡 | 11-12 | 3.0-3.5% | 80-6-5-5-4 | 中期趨勢 |
| 保守 | 13-15 | 4.0-5.0% | 85-5-4-3-3 | 長期趨勢 |
| 極端保守 | 16-20 | 5.0%+ | 90-4-3-2-1 | 低頻交易 |

---

## 重新訓練

### 使用新參數重新訓練

```bash
# 刪除舊模型
rm -r models/trained/BTCUSDT/15m

# 使用新參數訓練
python train.py \
  --symbol BTCUSDT \
  --timeframe 15m \
  --zigzag_depth 9 \
  --zigzag_deviation 2.5 \
  --epochs 100 \
  --batch_size 32
```

### 進行推理

```bash
python infer.py --symbol BTCUSDT --timeframe 15m
```

---

## 參考資源

- [Dealing with Imbalanced Data](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-machine-learning/)
- [Class Weights in Keras](https://keras.io/guides/training_keras_models/)
- [ZigZag Indicator Documentation](https://school.stockcharts.com/doku.php?id=technical_indicators:zigzag)
- [F1-Score vs Accuracy](https://en.wikipedia.org/wiki/F-score)

---

**最後更新**: 2026-01-07
**作者**: ZigZag Prediction Team
