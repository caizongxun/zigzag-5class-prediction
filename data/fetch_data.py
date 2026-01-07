import os
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import json

class CryptoDataFetcher:
    """Download OHLCV data and symbol-specific classifiers from Hugging Face."""
    
    DATASET_ID = 'zongowo111/v2-crypto-ohlcv-data'
    BASE_URL = 'https://huggingface.co/datasets/{}/resolve/main'.format(DATASET_ID)
    
    @staticmethod
    def fetch_symbol_timeframe(symbol: str = 'BTCUSDT', timeframe: str = '15m'):
        """Download OHLCV data for specific symbol and timeframe.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            timeframe: '15m' or '1h'
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        print(f'Fetching {symbol}_{timeframe} data...')
        
        try:
            # Download dataset from Hugging Face
            dataset = load_dataset(
                CryptoDataFetcher.DATASET_ID,
                data_files=f'ohlcv/{symbol}/{timeframe}.parquet',
                split='train'
            )
            
            df = dataset.to_pandas()
            
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            print(f'{symbol}_{timeframe} shape: {df.shape}')
            print(f'Columns: {list(df.columns)}')
            print(f'Date range: {df["timestamp"].min()} to {df["timestamp"].max()}')
            
            return df
            
        except Exception as e:
            print(f'Error fetching {symbol}_{timeframe}: {e}')
            return None
    
    @staticmethod
    def download_classifier(symbol: str, timeframe: str, output_dir: str = './models/trained'):
        """Download symbol and timeframe-specific classifier.
        
        CRITICAL: This is NOT a separate Stage1 classifier.
        Each symbol/timeframe has its OWN classification model in v1_model/SYMBOL/TIMEFRAME/
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            timeframe: '15m' or '1h'
            output_dir: Where to save the classifier
            
        Returns:
            Tuple of (model_path, params_path) or (None, None) if failed
        """
        print(f'\nDownloading {symbol} {timeframe} Classifier...')
        
        output_path = Path(output_dir) / symbol / timeframe
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Classification model
            model_file = f'v1_model/{symbol}/{timeframe}/classification.h5'
            params_file = f'v1_model/{symbol}/{timeframe}/params.json'
            
            model_url = f'{CryptoDataFetcher.BASE_URL}/{model_file}'
            params_url = f'{CryptoDataFetcher.BASE_URL}/{params_file}'
            
            print(f'  Downloading classification.h5...')
            model_path = output_path / 'classification.h5'
            hf_hub_download(
                repo_id=CryptoDataFetcher.DATASET_ID,
                filename=model_file,
                repo_type='dataset',
                local_dir='./models/trained',
                local_dir_use_symlinks=False,
                force_filename=str(model_path)
            )
            
            print(f'  Downloading params.json...')
            params_path = output_path / 'params.json'
            hf_hub_download(
                repo_id=CryptoDataFetcher.DATASET_ID,
                filename=params_file,
                repo_type='dataset',
                local_dir='./models/trained',
                local_dir_use_symlinks=False,
                force_filename=str(params_path)
            )
            
            print(f'  ✓ {symbol}_{timeframe} classifier downloaded successfully')
            print(f'    Model: {model_path}')
            print(f'    Params: {params_path}')
            
            return model_path, params_path
            
        except Exception as e:
            print(f'  Error downloading {symbol}_{timeframe} classifier: {e}')
            return None, None
    
    @staticmethod
    def load_classifier(symbol: str, timeframe: str, model_dir: str = './models/trained'):
        """Load downloaded classifier.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            timeframe: '15m' or '1h'
            model_dir: Directory containing classifiers
            
        Returns:
            Tuple of (model, params) or (None, None)
        """
        try:
            import tensorflow as tf
            
            model_path = Path(model_dir) / symbol / timeframe / 'classification.h5'
            params_path = Path(model_dir) / symbol / timeframe / 'params.json'
            
            if not model_path.exists():
                print(f'Model not found: {model_path}')
                return None, None
            
            # Load model
            model = tf.keras.models.load_model(model_path)
            
            # Load params
            with open(params_path, 'r') as f:
                params = json.load(f)
            
            print(f'Loaded {symbol}_{timeframe} classifier')
            print(f'  Accuracy: {params.get("accuracy", "N/A")}')
            print(f'  F1-Score: {params.get("f1_score", "N/A")}')
            
            return model, params
            
        except Exception as e:
            print(f'Error loading classifier: {e}')
            return None, None

if __name__ == '__main__':
    # Example: Download BTC 15m data and classifier
    print('=== Downloading Data and Classifier ===')
    
    # Download data
    df = CryptoDataFetcher.fetch_symbol_timeframe('BTCUSDT', '15m')
    
    if df is not None:
        print(f'Data downloaded: {len(df)} candles')
    
    # Download classifier for this symbol/timeframe
    model_path, params_path = CryptoDataFetcher.download_classifier('BTCUSDT', '15m')
    
    if model_path:
        print(f'\nClassifier files ready for training')
        
        # Load and verify
        model, params = CryptoDataFetcher.load_classifier('BTCUSDT', '15m')
        if model:
            print('✓ Classifier loaded successfully')
    else:
        print('✗ Failed to download classifier')
