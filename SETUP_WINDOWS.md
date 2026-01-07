# Windows 安裝指南 (Simplified Setup)

## 最快的方式 (5 分鐘)

### 第 1 步：下載並解壓

```bash
git clone https://github.com/caizongxun/zigzag-5class-prediction
cd zigzag-5class-prediction
```

### 第 2 步：創建虛擬環境

```bash
python -m venv venv
.\venv\Scripts\activate
```

確認提示符變成 `(venv) C:\...>`

### 第 3 步：安裝依賴 (新版 - 無 Jupyter Lab 問題)

```bash
pip install -r requirements.txt --no-cache-dir
```

**安裝時間:** 5-15 分鐘 (取決於網速)

### 第 4 步：驗證安裝

```bash
python tests/test_zigzag.py
```

**預期輸出:**
```
Running tests...

Test 1: ZigZag basic functionality
  Shape: (100, 6)
  Columns: [...]
  Label distribution: {...}
  Success!

Test 2: Feature engineering
  Shape: (200, 75)
  Number of features: 75
  Sample features: [...]
  Missing values: 0
  Success!

All tests passed!
```

如果看到 "All tests passed!" 則成功！

---

## 如果卡住了怎麼辦

### 錯誤 1: "OSError: No such file or directory"

**這是 Jupyter Lab 擴展問題 (已修複)**

使用新的 requirements.txt：

```bash
# 清空
deactivate
rmdir /s /q .venv

# 重新建立
python -m venv venv
.\venv\Scripts\activate

# 拉取最新 requirements
git pull origin main

# 重新安裝
pip install -r requirements.txt --no-cache-dir
```

### 錯誤 2: "pip install 非常慢或超時"

```bash
# 使用清華鏡像 (快 10 倍)
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 然後重新安裝
pip install -r requirements.txt --no-cache-dir

# 恢復官方源
pip config unset global.index-url
```

### 錯誤 3: "ModuleNotFoundError: No module named 'tensorflow'"

```bash
# 確認虛擬環境已激活
# 提示符應該有 (venv) 前綴

# 如果沒有，激活它
.\venv\Scripts\activate

# 重新安裝
pip install tensorflow==2.14.0 keras==2.14.0
```

### 錯誤 4: "CUDA not found"

**這是正常的。** 程序會自動使用 CPU。

不用修複，除非你想用 GPU 加速 (可選)。

---

## 運行 Jupyter Notebooks

### 方法 A: 使用 Notebook (推薦)

```bash
# 虛擬環境已激活的情況下
jupyter notebook
```

瀏覽器會自動打開 `http://localhost:8888`

導航到 `notebooks/` 文件夾

按順序運行:
1. `01_data_exploration.ipynb`
2. `02_feature_engineering.ipynb`
3. `03_model_training.ipynb`

### 方法 B: 使用命令行訓練 (更簡單)

```bash
# 訓練模型 (10-30 分鐘)
python train.py --epochs 100 --batch_size 32

# 運行推理
python infer.py --model ./models/trained
```

---

## 常見問題

### Q: 訓練為什麼這麼慢?

**A:** CPU 上正常 (30-60 分鐘)。

- 有 NVIDIA GPU: 2-5 分鐘
- 沒有 GPU: 30-60 分鐘
- Apple M1/M2: 5-10 分鐘

### Q: 可以中斷訓練嗎?

**A:** 可以，按 `Ctrl+C`。模型會保存進度。

### Q: 訓練到一半斷電了怎麼辦?

**A:** 刪除部分訓練的模型，重新開始:

```bash
rmdir /s /q models\trained
python train.py
```

### Q: 需要 GPU 嗎?

**A:** 不需要。CPU 也能工作，只是慢一點。

### Q: 如何使用 GPU?

**A:** 需要安裝 NVIDIA 的 CUDA (複雜)。對初學者不推薦。

---

## 數據來源

所有數據自動從 Hugging Face 下載：

- **數據集**: `zongowo111/v2-crypto-ohlcv-data`
- **Stage 1 分類器**: `zongowo111/BB-Bounce-Validity-Classifier-Stage1`

首次下載需要 5 分鐘，之後自動快取。

---

## 驗收標準

訓練完成後，模型應該達到：

| 指標 | 目標 | 預期 |
|------|------|------|
| 測試準確率 | >70% | 60-70% |
| 訓練時間 | <1 小時 | 30-60 分鐘 (CPU) |
| 推理延遲 | <10ms | <10ms |
| 模型大小 | <500MB | ~50MB |

---

## 下一步

1. **理解模型**: 閱讀 `README.md`
2. **調整參數**: 編輯 `config.yaml`
3. **改進準確率**: 見 `TROUBLESHOOTING.md` 中 "Low Accuracy" 部分
4. **部署上線**: 見 `DEPLOYMENT_GUIDE.md`

---

## 遇到問題?

見 `TROUBLESHOOTING.md` 尋找你的錯誤信息。

或在 GitHub 發 issue：
https://github.com/caizongxun/zigzag-5class-prediction/issues

---

**Happy Trading!**
