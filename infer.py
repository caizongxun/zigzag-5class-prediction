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
    
    # If not found, return default config
    print('  ⚠️  train_params.json not found, using fallback configuration')
    print('  Please run: python train.py --symbol BTCUSDT --timeframe 15m')
    return None

def load_model_fallback(model_dir):
    """Load model with fallback options"""
    model_dir = Path(model_dir)
    
    # Try to load from h5 file
    h5_files = list(model_dir.glob('*.h5')) + list(model_dir.glob('**/lstm_model.h5'))
    
    if h5_files:
        model_path = h5_files[0]
        print(f'  Loading model from: {model_path}')
        return keras.models.load_model(str(model_path))
    
    print(f'ERROR: Could not find model file (.h5)')
    return None

def main():
    parser = argparse.ArgumentParser(description='Make predictions with ZigZag Model')
    parser.add_argument('--model', default='./models/trained', help='Model directory')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--timeframe', default='15m', help='Timeframe')
    parser.add_argument('--n_candles', type=int, default=10, help='Number of recent candles to show')
    args = parser.parse_args()
    
    model_dir = Path(args.model) / args.symbol / args.timeframe
    
    print('Step 1: Loading Model Configuration...')
    train_params = find_and_load_config(model_dir)
    
    if train_params is None:
        print(f'\n⚠️  SETUP ERROR')
        print(f'\nNo trained model found for {args.symbol} {args.timeframe}')
        print(f'\nSolution: Train a model first')
        print(f'\n$ python train.py --symbol {args.symbol} --timeframe {args.timeframe} --epochs 50')
        print(f'\nOr try other symbols:')
        print(f'$ python train.py --symbol ETHUSDT --timeframe 15m')
        return
    
    feature_cols = train_params.get('feature_columns', [])
    print(f'  Features: {len(feature_cols)}')
    print(f'  Timesteps: {train_params.get("timesteps", 60)}')
    print(f'  Classes: {train_params.get("n_classes", 5)}')
    
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
    
    print(f'  Data shape: {df.shape}')
    print(f'  Date range: {df["timestamp"].min()} to {df["timestamp"].max()}')
    
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
    
    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in df.columns:
            print(f'  WARNING: Feature column not found: {col}')
            df[col] = 0
    
    # Fill NaN values
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill').fillna(0)
    
    print(f'  ✓ Features computed')
    
    print('\nStep 6: Making Predictions...')
    
    # Get data for normalization
    mean = np.array(train_params.get('mean', [0] * len(feature_cols)))
    std = np.array(train_params.get('std', [1] * len(feature_cols)))
    
    X = df[feature_cols].values
    X_norm = (X - mean) / (std + 1e-8)
    
    timesteps = train_params.get('timesteps', 60)
    
    # Get last timesteps candles
    if len(X_norm) >= timesteps:
        X_input = X_norm[-timesteps:].reshape(1, timesteps, -1)
        
        signal_classes, confidences = model.predict(X_input)
        signal_id = signal_classes[0]
        signal_name = ZigZagIndicator.get_label_name(signal_id)
        confidence = confidences[0]
        
        print(f'\n' + '='*60)
        print(f'LATEST PREDICTION')
        print(f'='*60)
        print(f'  Signal: {signal_name} (ID: {signal_id})')
        print(f'  Confidence: {confidence:.2%}')
        print(f'  Timestamp: {df.iloc[-1]["timestamp"]}')
        print(f'  Price: ${df.iloc[-1]["close"]:.2f}')
        print(f'  High: ${df.iloc[-1]["high"]:.2f}')
        print(f'  Low: ${df.iloc[-1]["low"]:.2f}')
        print(f'  Volume: {df.iloc[-1]["volume"]:.0f}')
        print(f'='*60)
    else:
        print(f'  ERROR: Not enough data. Need {timesteps} candles, have {len(X_norm)}')
        return
    
    # Show recent candles
    print(f'\nLAST {args.n_candles} CANDLES')
    print('-' * 80)
    print('Idx | Timestamp           | Open     | High     | Low      | Close    | Signal')
    print('-' * 80)
    
    start_idx = max(0, len(df) - args.n_candles)
    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        signal_label = zigzag.get_label_name(row['zigzag_label'])
        time_str = str(row.get('timestamp', 'N/A'))[:19]
        idx_display = i - start_idx
        print(f'{idx_display:3d} | {time_str} | {row["open"]:8.2f} | {row["high"]:8.2f} | {row["low"]:8.2f} | {row["close"]:8.2f} | {signal_label}')
    
    print(f'\n✓ Inference complete!')
    print(f'\nModel Information:')
    print(f'  Symbol: {train_params.get("symbol", args.symbol)}')
    print(f'  Timeframe: {train_params.get("timeframe", args.timeframe)}')
    print(f'  Total features: {train_params.get("n_features", len(feature_cols))}')
    print(f'  Model location: {model_dir}')
    print(f'\nSignal Meanings:')
    print(f'  NO_SIGNAL (0): No clear structure')
    print(f'  HH (1): Higher High - Bullish continuation')
    print(f'  LH (2): Lower High - Bearish divergence')
    print(f'  HL (3): Higher Low - Bearish reversal')
    print(f'  LL (4): Lower Low - Bearish continuation')

if __name__ == '__main__':
    main()
