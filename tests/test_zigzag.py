import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.zigzag_indicator import ZigZagIndicator
from src.features import FeatureEngineer

def test_zigzag_basic():
    print('Test 1: ZigZag basic functionality')
    
    # Create sample OHLC data
    dates = pd.date_range('2024-01-01', periods=100, freq='15min')
    np.random.seed(42)
    close_prices = 50000 + np.cumsum(np.random.randn(100) * 100)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices + np.random.randn(100) * 10,
        'high': close_prices + abs(np.random.randn(100) * 20),
        'low': close_prices - abs(np.random.randn(100) * 20),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # Apply ZigZag
    zigzag = ZigZagIndicator(depth=12, deviation=5, backstep=2)
    df = zigzag.label_kbars(df)
    
    print(f'  Shape: {df.shape}')
    print(f'  Columns: {list(df.columns)}')
    
    label_counts = df['zigzag_label'].value_counts()
    print(f'  Label distribution: {dict(label_counts)}')
    
    assert 'zigzag_label' in df.columns
    assert len(df) == 100
    print('  Success!\n')

def test_feature_engineering():
    print('Test 2: Feature engineering')
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=200, freq='15min')
    np.random.seed(42)
    close_prices = 50000 + np.cumsum(np.random.randn(200) * 100)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices + np.random.randn(200) * 10,
        'high': close_prices + abs(np.random.randn(200) * 20),
        'low': close_prices - abs(np.random.randn(200) * 20),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, 200),
        'zigzag_label': np.zeros(200, dtype=int)
    })
    
    # Feature engineering
    fe = FeatureEngineer(lookback_periods=[5, 10, 20, 50, 200])
    df = fe.calculate_all_features(df)
    
    feature_cols = fe.get_feature_columns(df)
    
    print(f'  Shape: {df.shape}')
    print(f'  Number of features: {len(feature_cols)}')
    print(f'  Sample features: {feature_cols[:10]}')
    
    missing = df[feature_cols].isnull().sum().sum()
    print(f'  Missing values: {missing}')
    
    assert len(feature_cols) > 50
    assert missing < len(df) * len(feature_cols) * 0.1
    print('  Success!\n')

if __name__ == '__main__':
    print('Running tests...\n')
    test_zigzag_basic()
    test_feature_engineering()
    print('All tests passed!')
