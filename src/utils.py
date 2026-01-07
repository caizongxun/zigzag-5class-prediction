import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import json

def print_data_info(df: pd.DataFrame, name: str = 'Data'):
    print(f'\n{name} Information:')
    print(f'Shape: {df.shape}')
    print(f'Columns: {list(df.columns)}')
    if 'timestamp' in df.columns:
        print(f'Date range: {df["timestamp"].min()} to {df["timestamp"].max()}')
    print(f'Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB')
    print(f'Null values: {df.isnull().sum().sum()}')

def time_series_split(df: pd.DataFrame, train_ratio=0.7, validation_ratio=0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + validation_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    return train_df, val_df, test_df

def normalize_data(data: np.ndarray, method='zscore') -> Tuple[np.ndarray, Dict]:
    if method == 'zscore':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        normalized = (data - mean) / (std + 1e-8)
        params = {'mean': mean, 'std': std, 'method': 'zscore'}
    elif method == 'minmax':
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        normalized = (data - data_min) / (data_max - data_min + 1e-8)
        params = {'min': data_min, 'max': data_max, 'method': 'minmax'}
    else:
        raise ValueError(f'Unknown normalization method: {method}')
    
    return normalized, params

def save_config(config_dict: Dict, save_path: str):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    print(f'Config saved to {save_path}')

def load_config(config_path: str) -> Dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)
