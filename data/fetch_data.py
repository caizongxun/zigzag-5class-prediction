import os
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download
import json

class CryptoDataFetcher:
    """Download OHLCV data and symbol-specific classifiers from Hugging Face."""
    
    DATASET_ID = 'zongowo111/v2-crypto-ohlcv-data'
    BASE_URL = 'https://huggingface.co/datasets/{}/resolve/main'.format(DATASET_ID)
    
    @staticmethod
    def fetch_symbol_timeframe(symbol: str = 'BTCUSDT', timeframe: str = '15m', cache_dir: str = './data/raw'):
        """Download OHLCV data for specific symbol and timeframe.
        
        IMPORTANT: The HF dataset uses a flat naming structure:
        BTC15m.parquet, BTC1h.parquet, ETH15m.parquet, etc.
        NOT: ohlcv/BTCUSDT/15m.parquet
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT' or 'BTC')
            timeframe: '15m' or '1h'
            cache_dir: Directory to cache parquet files
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        print(f'Fetching {symbol}_{timeframe} data...')
        
        try:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Try different file naming conventions used in the dataset
            # Convention 1: BTCUSDT_15m.parquet (underscore)
            # Convention 2: BTC15m.parquet (abbreviated without underscore)
            # Convention 3: ohlcv/BTCUSDT/15m.parquet (nested)
            
            file_attempts = [
                f'{symbol}_{timeframe}.parquet',  # BTCUSDT_15m.parquet
                f'{symbol}{timeframe}.parquet',   # BTCUSDT15m.parquet
                f'{symbol.replace("USDT", "")}{timeframe}.parquet',  # BTC15m.parquet
                f'ohlcv/{symbol}/{timeframe}.parquet',  # nested structure
            ]
            
            for file_path in file_attempts:
                try:
                    print(f'  Attempting: {file_path}...')
                    # Download from Hugging Face using hf_hub_download
                    local_path = hf_hub_download(
                        repo_id=CryptoDataFetcher.DATASET_ID,
                        filename=file_path,
                        repo_type='dataset',
                        cache_dir=str(cache_dir),
                        force_download=False,
                        timeout=30
                    )
                    
                    print(f'  Successfully found: {file_path}')
                    # Load parquet file
                    print(f'  Loading parquet file...')
                    df = pd.read_parquet(local_path)
                    
                    # Convert timestamp to datetime if it exists
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    print(f'{symbol}_{timeframe} shape: {df.shape}')
                    print(f'Columns: {list(df.columns)}')
                    if 'timestamp' in df.columns:
                        print(f'Date range: {df["timestamp"].min()} to {df["timestamp"].max()}')
                    
                    return df
                    
                except Exception as e:
                    continue
            
            # If all attempts failed
            print(f'Error: Could not find {symbol}_{timeframe} in dataset')
            print(f'  Tried paths: {file_attempts}')
            print(f'  Please verify symbol and timeframe are correct')
            return None
            
        except Exception as e:
            print(f'Error fetching {symbol}_{timeframe}: {type(e).__name__}: {str(e)[:200]}')
            return None
    
    @staticmethod
    def download_classifier(symbol: str, timeframe: str, output_dir: str = './models/trained'):
        """Download symbol and timeframe-specific classifier.
        
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
            # Classification model file paths in HF dataset
            model_file = f'v1_model/{symbol}/{timeframe}/classification.h5'
            params_file = f'v1_model/{symbol}/{timeframe}/params.json'
            
            # Download model
            print(f'  Downloading classification.h5...')
            model_path = hf_hub_download(
                repo_id=CryptoDataFetcher.DATASET_ID,
                filename=model_file,
                repo_type='dataset',
                cache_dir=str(output_path.parent.parent),
                force_download=False,
                timeout=60
            )
            
            # Copy to expected location
            import shutil
            final_model_path = output_path / 'classification.h5'
            shutil.copy(model_path, final_model_path)
            print(f'    Saved to: {final_model_path}')
            
            # Download params
            print(f'  Downloading params.json...')
            params_path_dl = hf_hub_download(
                repo_id=CryptoDataFetcher.DATASET_ID,
                filename=params_file,
                repo_type='dataset',
                cache_dir=str(output_path.parent.parent),
                force_download=False,
                timeout=30
            )
            
            # Copy to expected location
            final_params_path = output_path / 'params.json'
            shutil.copy(params_path_dl, final_params_path)
            print(f'    Saved to: {final_params_path}')
            
            print(f'  ✓ {symbol}_{timeframe} classifier downloaded successfully')
            print(f'    Model: {final_model_path}')
            print(f'    Params: {final_params_path}')
            
            return final_model_path, final_params_path
            
        except Exception as e:
            print(f'  Error downloading {symbol}_{timeframe} classifier: {type(e).__name__}')
            print(f'  Message: {str(e)[:200]}')
            print(f'  Note: Classifier might not exist for this symbol/timeframe combination')
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
            print(f'Loading TensorFlow model: {model_path}')
            model = tf.keras.models.load_model(model_path)
            
            # Load params
            with open(params_path, 'r') as f:
                params = json.load(f)
            
            print(f'Loaded {symbol}_{timeframe} classifier')
            print(f'  Accuracy: {params.get("accuracy", "N/A")}')
            print(f'  F1-Score: {params.get("f1_score", "N/A")}')
            
            return model, params
            
        except Exception as e:
            print(f'Error loading classifier: {type(e).__name__}: {str(e)}')
            return None, None

if __name__ == '__main__':
    # Example: Download BTC 15m data and classifier
    print('=== Downloading Data and Classifier ===')
    
    # Download data - try BTC instead of BTCUSDT
    df = CryptoDataFetcher.fetch_symbol_timeframe('BTCUSDT', '15m')
    
    if df is None:
        print('\nTrying alternative symbol format (BTC instead of BTCUSDT)...')
        df = CryptoDataFetcher.fetch_symbol_timeframe('BTC', '15m')
    
    if df is not None:
        print(f'\nData downloaded successfully: {len(df)} candles')
    else:
        print('Failed to download data - check symbol and timeframe')
    
    # Download classifier for this symbol/timeframe
    model_path, params_path = CryptoDataFetcher.download_classifier('BTCUSDT', '15m')
    
    if model_path:
        print(f'\nClassifier files ready for training')
        
        # Load and verify
        model, params = CryptoDataFetcher.load_classifier('BTCUSDT', '15m')
        if model:
            print('✓ Classifier loaded successfully')
        else:
            print('✗ Failed to load classifier')
    else:
        print('✗ Failed to download classifier')
