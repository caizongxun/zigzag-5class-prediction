import sys
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
from data.fetch_data import CryptoDataFetcher
from src.zigzag_indicator import ZigZagIndicator
from src.features import FeatureEngineer
from src.models import LSTMXGBoostModel
from src.utils import load_config
import json
import pickle
from tensorflow import keras

def find_and_load_config(model_dir):
    """Search for train_params.json in multiple locations"""
    
    search_paths = [
        Path(model_dir) / 'train_params.json',
        Path(model_dir).parent / 'train_params.json',
        Path(model_dir).parent.parent / 'train_params.json',
        Path('./models/trained') / 'train_params.json',
        Path('./models') / 'train_params.json',
    ]
    
    for config_path in search_paths:
        if config_path.exists():
            print(f'  Found config at: {config_path}')
            with open(config_path, 'r') as f:
                return json.load(f)
    
    print('  ⚠️  train_params.json not found, using fallback configuration')
    return None

def load_model_fallback(model_dir):
    """Load model with fallback options"""
    model_dir = Path(model_dir)
    
    h5_files = list(model_dir.glob('*.h5')) + list(model_dir.glob('**/lstm_model.h5'))
    
    if h5_files:
        model_path = h5_files[0]
        print(f'  Loading model from: {model_path}')
        return keras.models.load_model(str(model_path))
    
    print(f'ERROR: Could not find model file (.h5)')
    return None

def main():
    parser = argparse.ArgumentParser(description='Make predictions with confidence threshold')
    parser.add_argument('--model', default='./models/trained', help='Model directory')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--timeframe', default='15m', help='Timeframe')
    parser.add_argument('--threshold', type=float, default=0.60, help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--n_candles', type=int, default=10, help='Number of recent candles to show')
    args = parser.parse_args()
    
    model_dir = Path(args.model) / args.symbol / args.timeframe
    
    print('='*70)
    print('ZIGZAG PREDICTION WITH CONFIDENCE THRESHOLD')
    print('='*70)
    print(f'\nConfiguration:')
    print(f'  Symbol: {args.symbol}')
    print(f'  Timeframe: {args.timeframe}')
    print(f'  Confidence Threshold: {args.threshold:.0%}')
    print(f'  Model Dir: {model_dir}')
    
    print('\nStep 1: Loading Model Configuration...')
    train_params = find_and_load_config(model_dir)
    
    if train_params is None:
        print(f'\n⚠️  SETUP ERROR')
        print(f'\nNo trained model found for {args.symbol} {args.timeframe}')
        print(f'\nPlease run: python train.py --symbol {args.symbol} --timeframe {args.timeframe}')
        return
    
    feature_cols = train_params.get('feature_columns', [])
    print(f'  ✓ Features: {len(feature_cols)}')
    print(f'  ✓ Timesteps: {train_params.get("timesteps", 60)}')
    print(f'  ✓ Classes: {train_params.get("n_classes", 5)}')
    
    print('\nStep 2: Loading LSTM Model...')
    try:
        model = LSTMXGBoostModel(
            timesteps=train_params.get('timesteps', 60),
            n_features=len(feature_cols),
            n_classes=train_params.get('n_classes', 5)
        )
        model.load(str(model_dir))
        print(f'  ✓ Model loaded successfully')
    except Exception as e:
        print(f'  ERROR loading model: {e}')
        print(f'  Please retrain: python train.py --symbol {args.symbol} --timeframe {args.timeframe}')
        return
    
    print('\nStep 3: Fetching Latest Data...')
    fetcher = CryptoDataFetcher()
    df = fetcher.fetch_symbol_timeframe(args.symbol, args.timeframe)
    
    if df is None:
        print(f'ERROR: Could not fetch {args.symbol} {args.timeframe}')
        return
    
    print(f'  ✓ Data shape: {df.shape}')
    print(f'  ✓ Date range: {df["timestamp"].min()} to {df["timestamp"].max()}')
    
    print('\nStep 4: Applying ZigZag Indicator...')
    zigzag_config = train_params.get('zigzag_config', {
        'depth': 10,
        'deviation': 2.5,
        'backstep': 2
    })
    zigzag = ZigZagIndicator(
        depth=zigzag_config.get('depth', 10),
        deviation=zigzag_config.get('deviation', 2.5),
        backstep=zigzag_config.get('backstep', 2)
    )
    df = zigzag.label_kbars(df)
    print(f'  ✓ ZigZag labels applied')
    
    print('\nStep 5: Feature Engineering...')
    fe = FeatureEngineer()
    df = fe.calculate_all_features(df)
    
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill').fillna(0)
    
    print(f'  ✓ Features computed')
    
    print('\nStep 6: Making Predictions with Confidence...')
    
    mean = np.array(train_params.get('mean', [0] * len(feature_cols)))
    std = np.array(train_params.get('std', [1] * len(feature_cols)))
    
    X = df[feature_cols].values
    X_norm = (X - mean) / (std + 1e-8)
    
    timesteps = train_params.get('timesteps', 60)
    
    if len(X_norm) >= timesteps:
        X_input = X_norm[-timesteps:].reshape(1, timesteps, -1)
        
        # 獲取原始預測概率
        raw_predictions = model.lstm_model.predict(X_input, verbose=0)
        probabilities = raw_predictions[0]  # 5 個類別的概率
        
        # 找到最高概率的類別
        top_signal_id = np.argmax(probabilities)
        top_confidence = probabilities[top_signal_id]
        top_signal_name = ZigZagIndicator.get_label_name(top_signal_id)
        
        # 找到次高概率
        sorted_indices = np.argsort(probabilities)[::-1]  # 降序排列
        second_signal_id = sorted_indices[1]
        second_confidence = probabilities[second_signal_id]
        second_signal_name = ZigZagIndicator.get_label_name(second_signal_id)
        
        print(f'\n' + '='*70)
        print(f'RAW PREDICTIONS (所有類別概率)')
        print(f'='*70)
        
        # 顯示所有類別的概率
        for i in range(5):
            label_name = ZigZagIndicator.get_label_name(i)
            prob = probabilities[i]
            bar_length = int(prob * 50)
            bar = '█' * bar_length
            print(f'  {label_name:12} ({i}): {prob:6.2%} {bar}')
        
        print(f'\n' + '='*70)
        print(f'SIGNAL DECISION')
        print(f'='*70)
        print(f'  Top Signal: {top_signal_name} (ID: {top_signal_id})')
        print(f'  Confidence: {top_confidence:.2%}')
        print(f'  Second: {second_signal_name} ({second_confidence:.2%})')
        print(f'  Threshold: {args.threshold:.0%}')
        
        # 根據閾值進行決策
        if top_confidence >= args.threshold:
            trade_decision = top_signal_name
            decision_status = '✅ STRONG SIGNAL'
        else:
            trade_decision = 'NO_SIGNAL'
            decision_status = '⚠️  WEAK SIGNAL'
        
        print(f'\n  Decision: {decision_status}')
        print(f'  Trade Signal: {trade_decision}')
        print(f'='*70)
        
        # 交易信息
        print(f'\nLatest Candle Information:')
        print(f'  Timestamp: {df.iloc[-1]["timestamp"]}')
        print(f'  Open: ${df.iloc[-1]["open"]:.2f}')
        print(f'  High: ${df.iloc[-1]["high"]:.2f}')
        print(f'  Low: ${df.iloc[-1]["low"]:.2f}')
        print(f'  Close: ${df.iloc[-1]["close"]:.2f}')
        print(f'  Volume: {df.iloc[-1]["volume"]:.0f}')
        
    else:
        print(f'  ERROR: Not enough data. Need {timesteps} candles, have {len(X_norm)}')
        return
    
    # 顯示最近的 K 線
    print(f'\nLAST {args.n_candles} CANDLES')
    print('-' * 90)
    print('Idx | Timestamp           | Open     | High     | Low      | Close    | ZigZag Signal')
    print('-' * 90)
    
    start_idx = max(0, len(df) - args.n_candles)
    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        signal_label = zigzag.get_label_name(row['zigzag_label'])
        time_str = str(row.get('timestamp', 'N/A'))[:19]
        idx_display = i - start_idx
        print(f'{idx_display:3d} | {time_str} | {row["open"]:8.2f} | {row["high"]:8.2f} | {row["low"]:8.2f} | {row["close"]:8.2f} | {signal_label}')
    
    print(f'\n✓ Prediction complete!')
    print(f'\nSignal Meanings:')
    print(f'  NO_SIGNAL (0): 無明確結構')
    print(f'  HH (1): 更高的高 - 看漲延續 ✅')
    print(f'  LH (2): 更高的低 - 看漲反轉 ✅')
    print(f'  HL (3): 更低的高 - 看跌反轉 ⚠️')
    print(f'  LL (4): 更低的低 - 看跌延續 ❌')
    print(f'\nTip: Adjust threshold with --threshold 0.50 to 0.80')

if __name__ == '__main__':
    main()
