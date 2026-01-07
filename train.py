import sys
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
from data.fetch_data import CryptoDataFetcher
from src.zigzag_indicator import ZigZagIndicator
from src.features import FeatureEngineer
from src.models import LSTMXGBoostModel, VolatilityModel
from src.utils import time_series_split, normalize_data, print_data_info, save_config
from src.config import config

def main():
    parser = argparse.ArgumentParser(description='Train ZigZag 5-Class Model')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--timeframe', default='15m', help='Timeframe (15m, 1h)')
    parser.add_argument('--output', default='./models/trained', help='Model save directory')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print('Step 1: Downloading Stage 1 Classifier...')
    CryptoDataFetcher.download_stage1_classifier('lstm_validity')
    
    print('\nStep 2: Fetching Data...')
    fetcher = CryptoDataFetcher()
    df = fetcher.fetch_symbol_timeframe(args.symbol, args.timeframe)
    
    if df is None:
        print(f'Error: Could not fetch {args.symbol} {args.timeframe}')
        return
    
    print_data_info(df, f'{args.symbol}_{args.timeframe}')
    
    print('\nStep 3: Applying ZigZag Indicator...')
    zigzag = ZigZagIndicator(
        depth=config.get('zigzag.depth', 12),
        deviation=config.get('zigzag.deviation', 5),
        backstep=config.get('zigzag.backstep', 2)
    )
    df = zigzag.label_kbars(df)
    
    label_dist = df['zigzag_label'].value_counts().sort_index()
    print('\nLabel Distribution:')
    for label_id, count in label_dist.items():
        label_name = zigzag.get_label_name(label_id)
        print(f'  {label_name}: {count} ({count/len(df)*100:.2f}%)')
    
    print('\nStep 4: Feature Engineering...')
    fe = FeatureEngineer(lookback_periods=config.get('features.lookback_periods'))
    df = fe.calculate_all_features(df)
    
    feature_cols = fe.get_feature_columns(df)
    feature_cols = [col for col in feature_cols if col not in ['symbol', 'timestamp', 'date']]
    
    df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(0)
    
    print(f'Total features: {len(feature_cols)}')
    print(f'Sample features: {feature_cols[:10]}')
    
    print('\nStep 5: Time Series Split...')
    train_df, val_df, test_df = time_series_split(
        df,
        train_ratio=0.7,
        validation_ratio=0.15
    )
    print(f'Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}')
    
    print('\nStep 6: Preparing Data...')
    X_train = train_df[feature_cols].values
    y_train = train_df['zigzag_label'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['zigzag_label'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['zigzag_label'].values
    
    X_train_norm, train_params = normalize_data(X_train, method='zscore')
    X_val_norm = (X_val - train_params['mean']) / (train_params['std'] + 1e-8)
    X_test_norm = (X_test - train_params['mean']) / (train_params['std'] + 1e-8)
    
    print('\nStep 7: Creating Sequences...')
    model = LSTMXGBoostModel(timesteps=60, n_features=len(feature_cols), n_classes=5)
    X_train_seq, y_train_seq = model.create_sequences(X_train_norm, y_train)
    X_val_seq, y_val_seq = model.create_sequences(X_val_norm, y_val)
    X_test_seq, y_test_seq = model.create_sequences(X_test_norm, y_test)
    
    print(f'Sequences - Train: {X_train_seq.shape}, Val: {X_val_seq.shape}, Test: {X_test_seq.shape}')
    
    print('\nStep 8: Training Model...')
    history = model.train(
        X_train_seq, y_train_seq,
        X_val_seq, y_val_seq,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=10
    )
    
    print(f'Training complete!')
    print(f'LSTM Train Accuracy: {history["lstm_train_acc"]:.4f}')
    print(f'LSTM Val Accuracy: {history["lstm_val_acc"]:.4f}')
    
    print('\nStep 9: Evaluating Model...')
    metrics = model.evaluate(X_test_seq, y_test_seq)
    
    print(f'\nModel Evaluation:')
    print(f'Accuracy: {metrics["accuracy"]:.4f}')
    print(f'Precision: {metrics["precision"]:.4f}')
    print(f'Recall: {metrics["recall"]:.4f}')
    print(f'F1-Score: {metrics["f1"]:.4f}')
    
    print('\nStep 10: Training Volatility Model...')
    vol_model = VolatilityModel(atr_multiplier=1.5, atr_window=14)
    
    y_train_vol = vol_model.create_labels(train_df)
    X_train_vol = vol_model.create_features(train_df)
    
    y_val_vol = vol_model.create_labels(val_df)
    X_val_vol = vol_model.create_features(val_df)
    
    vol_metrics = vol_model.train(X_train_vol, y_train_vol, X_val_vol, y_val_vol)
    
    print(f'\nVolatility Model Metrics:')
    for key, value in vol_metrics.items():
        print(f'  {key}: {value:.4f}')
    
    print('\nStep 11: Saving Models...')
    model.save(output_dir)
    
    train_params['feature_columns'] = feature_cols
    train_params['zigzag_config'] = {
        'depth': zigzag.depth,
        'deviation': zigzag.deviation,
        'backstep': zigzag.backstep
    }
    train_params['model_metrics'] = metrics
    
    save_config(train_params, output_dir / 'train_params.json')
    
    print(f'\nAll models saved to {output_dir}')
    print('\nTraining complete! You can now use the model for inference.')

if __name__ == '__main__':
    main()
