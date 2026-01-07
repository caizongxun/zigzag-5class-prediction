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

def main():
    parser = argparse.ArgumentParser(description='Make predictions with ZigZag Model')
    parser.add_argument('--model', default='./models/trained', help='Model directory')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--timeframe', default='15m', help='Timeframe')
    parser.add_argument('--n_candles', type=int, default=10, help='Number of recent candles to show')
    args = parser.parse_args()
    
    model_dir = Path(args.model) / args.symbol / args.timeframe
    
    print('Step 1: Loading Model Configuration...')
    
    # Try loading from symbol-specific directory first, then fallback to root
    if (model_dir / 'train_params.json').exists():
        with open(model_dir / 'train_params.json', 'r') as f:
            train_params = json.load(f)
        print(f'  Loaded from: {model_dir / "train_params.json"}')
    elif (Path(args.model) / 'train_params.json').exists():
        with open(Path(args.model) / 'train_params.json', 'r') as f:
            train_params = json.load(f)
        print(f'  Loaded from: {Path(args.model) / "train_params.json"}')
    else:
        print(f'ERROR: Could not find train_params.json')
        print(f'  Searched in: {model_dir}')
        print(f'  Also searched in: {Path(args.model)}')
        return
    
    feature_cols = train_params['feature_columns']
    print(f'  Features: {len(feature_cols)}')
    print(f'  Timesteps: {train_params["timesteps"]}')
    
    print('\nStep 2: Loading LSTM Model...')
    model = LSTMXGBoostModel(
        timesteps=train_params['timesteps'],
        n_features=len(feature_cols),
        n_classes=train_params['n_classes']
    )
    model.load(str(model_dir))
    print(f'  Model loaded successfully')
    
    print('\nStep 3: Fetching Latest Data...')
    fetcher = CryptoDataFetcher()
    df = fetcher.fetch_symbol_timeframe(args.symbol, args.timeframe)
    
    if df is None:
        print(f'ERROR: Could not fetch {args.symbol} {args.timeframe}')
        return
    
    print(f'  Data shape: {df.shape}')
    print(f'  Date range: {df["timestamp"].min()} to {df["timestamp"].max()}')
    
    print('\nStep 4: Applying ZigZag Indicator...')
    zigzag_config = train_params['zigzag_config']
    zigzag = ZigZagIndicator(
        depth=zigzag_config['depth'],
        deviation=zigzag_config['deviation'],
        backstep=zigzag_config.get('backstep', 2)
    )
    df = zigzag.label_kbars(df)
    print(f'  ZigZag labels applied')
    
    print('\nStep 5: Feature Engineering...')
    fe = FeatureEngineer()
    df = fe.calculate_all_features(df)
    
    # Fill NaN values
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill').fillna(0)
    
    print(f'  Features computed')
    
    print('\nStep 6: Making Predictions...')
    
    # Get data for normalization
    mean = np.array(train_params['mean'])
    std = np.array(train_params['std'])
    
    X = df[feature_cols].values
    X_norm = (X - mean) / (std + 1e-8)
    
    # Get last timesteps candles
    if len(X_norm) >= train_params['timesteps']:
        X_input = X_norm[-train_params['timesteps']:].reshape(1, train_params['timesteps'], -1)
        
        signal_classes, confidences = model.predict(X_input)
        signal_id = signal_classes[0]
        signal_name = ZigZagIndicator.get_label_name(signal_id)
        confidence = confidences[0]
        
        print(f'\n=== LATEST PREDICTION ===')
        print(f'  Signal: {signal_name} (ID: {signal_id})')
        print(f'  Confidence: {confidence:.2%}')
        print(f'  Timestamp: {df.iloc[-1]["timestamp"] if "timestamp" in df.columns else "N/A"}')
        print(f'  Price: ${df.iloc[-1]["close"]:.2f}')
        print(f'  High: ${df.iloc[-1]["high"]:.2f}')
        print(f'  Low: ${df.iloc[-1]["low"]:.2f}')
        print(f'  Volume: {df.iloc[-1]["volume"]:.0f}')
    else:
        print(f'  ERROR: Not enough data. Need {train_params["timesteps"]} candles, have {len(X_norm)}')
        return
    
    # Show recent candles
    print(f'\n=== LAST {args.n_candles} CANDLES ===')
    print('Idx | Timestamp | Open | High | Low | Close | ZigZag Signal')
    print('-' * 75)
    
    start_idx = max(0, len(df) - args.n_candles)
    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        signal_label = zigzag.get_label_name(row['zigzag_label'])
        time_str = str(row.get('timestamp', 'N/A'))[:19]
        idx_display = i - (len(df) - args.n_candles)
        print(f'{idx_display:3d} | {time_str} | {row["open"]:7.2f} | {row["high"]:7.2f} | {row["low"]:7.2f} | {row["close"]:7.2f} | {signal_label}')
    
    print(f'\n✓ Inference complete!')
    print(f'\nModel Information:')
    print(f'  Symbol: {train_params["symbol"]}')
    print(f'  Timeframe: {train_params["timeframe"]}')
    print(f'  Total features: {train_params["n_features"]}')
    print(f'  Model location: {model_dir}')

if __name__ == '__main__':
    main()
