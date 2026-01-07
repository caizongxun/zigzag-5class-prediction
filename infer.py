import sys
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
from data.fetch_data import CryptoDataFetcher
from src.zigzag_indicator import ZigZagIndicator
from src.features import FeatureEngineer
from src.models import LSTMXGBoostModel
from src.utils import normalize_data, load_config

def main():
    parser = argparse.ArgumentParser(description='Make predictions with ZigZag Model')
    parser.add_argument('--model', default='./models/trained', help='Model directory')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--timeframe', default='15m', help='Timeframe')
    parser.add_argument('--n_candles', type=int, default=10, help='Number of recent candles to show')
    args = parser.parse_args()
    
    model_dir = Path(args.model)
    
    print('Step 1: Loading Model Configuration...')
    train_params = load_config(model_dir / 'train_params.json')
    feature_cols = train_params['feature_columns']
    
    print('Step 2: Loading Model...')
    model = LSTMXGBoostModel(timesteps=60, n_features=len(feature_cols), n_classes=5)
    model.load(model_dir)
    
    print('Step 3: Fetching Latest Data...')
    fetcher = CryptoDataFetcher()
    df = fetcher.fetch_symbol_timeframe(args.symbol, args.timeframe)
    
    if df is None:
        print(f'Error fetching {args.symbol} {args.timeframe}')
        return
    
    print('Step 4: Applying ZigZag Indicator...')
    zigzag = ZigZagIndicator(
        depth=train_params['zigzag_config']['depth'],
        deviation=train_params['zigzag_config']['deviation'],
        backstep=train_params['zigzag_config']['backstep']
    )
    df = zigzag.label_kbars(df)
    
    print('Step 5: Feature Engineering...')
    fe = FeatureEngineer()
    df = fe.calculate_all_features(df)
    df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(0)
    
    print('Step 6: Making Predictions...')
    mean = np.array(train_params['mean'])
    std = np.array(train_params['std'])
    
    X = df[feature_cols].values
    X_norm = (X - mean) / (std + 1e-8)
    
    # Get last 60 candles
    if len(X_norm) >= 60:
        X_input = X_norm[-60:].reshape(1, 60, -1)
        
        signal_classes, confidences = model.predict(X_input)
        signal_id = signal_classes[0]
        signal_name = ZigZagIndicator.get_label_name(signal_id)
        confidence = confidences[0]
        
        print(f'\nLatest Prediction:')
        print(f'Signal: {signal_name}')
        print(f'Confidence: {confidence:.2%}')
        print(f'Timestamp: {df.iloc[-1]["timestamp"] if "timestamp" in df.columns else "N/A"}')
        print(f'Price: {df.iloc[-1]["close"]:.2f}')
    else:
        print(f'Not enough data. Need 60 candles, have {len(X_norm)}')
        return
    
    print(f'\nLast {args.n_candles} Candles:')
    print('\nIdx | Time | Open | High | Low | Close | Signal')
    print('-' * 60)
    
    for i in range(max(0, len(df) - args.n_candles), len(df)):
        row = df.iloc[i]
        signal_label = zigzag.get_label_name(row['zigzag_label'])
        time_str = str(row.get('timestamp', 'N/A'))[:16]
        print(f'{i:3d} | {time_str} | {row["open"]:7.2f} | {row["high"]:7.2f} | {row["low"]:7.2f} | {row["close"]:7.2f} | {signal_label}')
    
    print(f'\nInference complete!')

if __name__ == '__main__':
    main()
