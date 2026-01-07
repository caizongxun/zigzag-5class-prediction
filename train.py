import sys
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
from data.fetch_data import CryptoDataFetcher
from src.zigzag_indicator import ZigZagIndicator
from src.features import FeatureEngineer
from src.models import LSTMXGBoostModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description='Train ZigZag 5-Class Model')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--timeframe', default='15m', help='Timeframe (15m, 1h)')
    parser.add_argument('--output', default='./models/trained', help='Model save directory')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--timesteps', type=int, default=60, help='Sequence length for LSTM')
    parser.add_argument('--zigzag_depth', type=int, default=12, help='ZigZag depth parameter')
    parser.add_argument('--zigzag_deviation', type=int, default=5, help='ZigZag deviation parameter (%)')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print('Step 1: Downloading Symbol-Specific Classifier...')
    print(f'  Symbol: {args.symbol}')
    print(f'  Timeframe: {args.timeframe}')
    print(f'  Note: Each symbol/timeframe has its OWN classification model')
    print(f'  Location: v1_model/{args.symbol}/{args.timeframe}/')
    
    model_path, params_path = CryptoDataFetcher.download_classifier(
        args.symbol, 
        args.timeframe,
        output_dir=str(output_dir)
    )
    
    if not model_path:
        print(f'ERROR: Could not download classifier for {args.symbol} {args.timeframe}')
        print(f'  Verify that {args.symbol} exists in the dataset')
        print(f'  Available symbols typically include: BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, SOLUSDT, etc.')
        return
    
    print('\nStep 2: Fetching Data...')
    fetcher = CryptoDataFetcher()
    df = fetcher.fetch_symbol_timeframe(args.symbol, args.timeframe)
    
    if df is None:
        print(f'Error: Could not fetch {args.symbol} {args.timeframe}')
        print(f'  This usually means the data file does not exist in the dataset')
        print(f'  Available symbols: BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, SOLUSDT, ...')
        print(f'  Available timeframes: 15m, 1h')
        return
    
    print(f'\nData Summary:')
    print(f'  Shape: {df.shape}')
    print(f'  Columns: {list(df.columns)}')
    if 'timestamp' in df.columns:
        print(f'  Date range: {df["timestamp"].min()} to {df["timestamp"].max()}')
    
    # ============================================
    # STEP 3: Applying ZigZag Indicator
    # ============================================
    print('\nStep 3: Applying ZigZag Indicator...')
    print(f'  Depth: {args.zigzag_depth}, Deviation: {args.zigzag_deviation}%')
    
    zigzag = ZigZagIndicator(depth=args.zigzag_depth, deviation=args.zigzag_deviation, backstep=2)
    df = zigzag.label_kbars(df)
    
    label_distribution = df['zigzag_label'].value_counts().sort_index()
    print(f'  ZigZag Labels:')
    for label_id, count in label_distribution.items():
        label_name = ZigZagIndicator.get_label_name(label_id)
        pct = (count / len(df)) * 100
        print(f'    {label_name} ({label_id}): {count} ({pct:.1f}%)')
    
    # ============================================
    # STEP 4: Feature Engineering
    # ============================================
    print('\nStep 4: Feature Engineering...')
    feature_engineer = FeatureEngineer()
    df = feature_engineer.calculate_all_features(df)
    
    feature_cols = FeatureEngineer.get_feature_columns(df)
    print(f'  Total features created: {len(feature_cols)}')
    print(f'  Sample features: {feature_cols[:10]}')
    
    # Handle NaN values
    df = df.fillna(method='bfill').fillna(method='ffill')
    print(f'  NaN values removed')
    
    # ============================================
    # STEP 5: Time Series Split
    # ============================================
    print('\nStep 5: Time Series Split...')
    n_samples = len(df)
    train_size = int(n_samples * 0.7)  # 70%
    val_size = int(n_samples * 0.15)   # 15%
    test_size = n_samples - train_size - val_size  # 15%
    
    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:train_size + val_size].copy()
    test_df = df.iloc[train_size + val_size:].copy()
    
    print(f'  Total samples: {n_samples}')
    print(f'  Train: {len(train_df)} ({len(train_df)/n_samples*100:.1f}%)')
    print(f'  Validation: {len(val_df)} ({len(val_df)/n_samples*100:.1f}%)')
    print(f'  Test: {len(test_df)} ({len(test_df)/n_samples*100:.1f}%)')
    
    # ============================================
    # STEP 6: Data Preparation & Normalization
    # ============================================
    print('\nStep 6: Data Preparation...')
    
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df['zigzag_label'].values.astype(np.int32)
    
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df['zigzag_label'].values.astype(np.int32)
    
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df['zigzag_label'].values.astype(np.int32)
    
    # Normalize features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f'  Features normalized using StandardScaler')
    print(f'  X_train shape: {X_train_scaled.shape}')
    print(f'  X_val shape: {X_val_scaled.shape}')
    print(f'  X_test shape: {X_test_scaled.shape}')
    print(f'  Feature mean: {X_train_scaled.mean():.6f}, std: {X_train_scaled.std():.6f}')
    
    # ============================================
    # STEP 7: Creating LSTM Sequences
    # ============================================
    print('\nStep 7: Creating Sequences for LSTM...')
    print(f'  Sequence length (timesteps): {args.timesteps}')
    
    model = LSTMXGBoostModel(
        timesteps=args.timesteps, 
        n_features=len(feature_cols), 
        n_classes=5
    )
    
    X_train_seq, y_train_seq = model.create_sequences(X_train_scaled, y_train)
    X_val_seq, y_val_seq = model.create_sequences(X_val_scaled, y_val)
    X_test_seq, y_test_seq = model.create_sequences(X_test_scaled, y_test)
    
    print(f'  Train sequences: {X_train_seq.shape}')
    print(f'  Val sequences: {X_val_seq.shape}')
    print(f'  Test sequences: {X_test_seq.shape}')
    print(f'  Class distribution in train: {np.unique(y_train_seq, return_counts=True)}')
    
    # ============================================
    # STEP 8: Training Model
    # ============================================
    print('\nStep 8: Training LSTM Model...')
    print(f'  Epochs: {args.epochs}')
    print(f'  Batch size: {args.batch_size}')
    print(f'  Early stopping patience: 10')
    
    train_results = model.train(
        X_train_seq, y_train_seq,
        X_val_seq, y_val_seq,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=10
    )
    
    print(f'\n  Training Results:')
    print(f'    LSTM Train Accuracy: {train_results["lstm_train_acc"]:.4f}')
    print(f'    LSTM Val Accuracy: {train_results["lstm_val_acc"]:.4f}')
    
    # ============================================
    # STEP 9: Evaluating Model
    # ============================================
    print('\nStep 9: Evaluating Model on Test Set...')
    
    test_results = model.evaluate(X_test_seq, y_test_seq)
    
    print(f'\n  Test Set Performance:')
    print(f'    Accuracy: {test_results["accuracy"]:.4f}')
    print(f'    Precision: {test_results["precision"]:.4f}')
    print(f'    Recall: {test_results["recall"]:.4f}')
    print(f'    F1-Score: {test_results["f1"]:.4f}')
    
    # Print confusion matrix
    print(f'\n  Confusion Matrix:')
    cm = test_results['confusion_matrix']
    label_names = [ZigZagIndicator.get_label_name(i) for i in range(5)]
    print(f'    {" ".join([f"{name:>10}" for name in label_names])}')
    for i, row in enumerate(cm):
        print(f'{label_names[i]:>3} {" ".join([f"{val:>10}" for val in row])}')
    
    # ============================================
    # STEP 10: Saving Models
    # ============================================
    print('\nStep 10: Saving Models...')
    
    symbol_model_dir = Path(args.output) / args.symbol / args.timeframe
    symbol_model_dir.mkdir(parents=True, exist_ok=True)
    
    model.save(str(symbol_model_dir))
    
    # Save scaler for later inference
    import pickle
    scaler_path = symbol_model_dir / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f'  Scaler saved to {scaler_path}')
    
    # Save feature columns
    feature_cols_path = symbol_model_dir / 'feature_columns.pkl'
    with open(feature_cols_path, 'wb') as f:
        pickle.dump(feature_cols, f)
    print(f'  Feature columns saved to {feature_cols_path}')
    
    # Save training metadata
    metadata = {
        'symbol': args.symbol,
        'timeframe': args.timeframe,
        'timesteps': args.timesteps,
        'n_features': len(feature_cols),
        'n_classes': 5,
        'zigzag_depth': args.zigzag_depth,
        'zigzag_deviation': args.zigzag_deviation,
        'test_accuracy': float(test_results['accuracy']),
        'test_f1': float(test_results['f1']),
        'train_samples': len(X_train_seq),
        'val_samples': len(X_val_seq),
        'test_samples': len(X_test_seq)
    }
    
    import json
    metadata_path = symbol_model_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f'  Metadata saved to {metadata_path}')
    
    # ============================================
    # Summary
    # ============================================
    print('\n' + '='*70)
    print('TRAINING COMPLETE')
    print('='*70)
    print(f'\nModel Information:')
    print(f'  Symbol: {args.symbol}')
    print(f'  Timeframe: {args.timeframe}')
    print(f'  Data range: {df["timestamp"].min()} to {df["timestamp"].max()}')
    print(f'  Total samples: {len(df)}')
    print(f'\nModel Performance:')
    print(f'  Test Accuracy: {test_results["accuracy"]:.4f}')
    print(f'  Test F1-Score: {test_results["f1"]:.4f}')
    print(f'\nModel Location: {symbol_model_dir}')
    print(f'  - LSTM Model: {symbol_model_dir / "lstm_model.h5"}')
    print(f'  - Config: {symbol_model_dir / "config.json"}')
    print(f'  - Scaler: {symbol_model_dir / "scaler.pkl"}')
    print(f'  - Features: {symbol_model_dir / "feature_columns.pkl"}')
    print(f'  - Metadata: {symbol_model_dir / "metadata.json"}')
    print(f'\nNext Steps:')
    print(f'  1. Use infer.py to make predictions on new data')
    print(f'  2. Evaluate on other symbols/timeframes')
    print(f'  3. Deploy model to production')
    print('='*70)

if __name__ == '__main__':
    main()
