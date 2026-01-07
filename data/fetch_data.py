import os
import pandas as pd
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download
from datasets import load_dataset
import pickle
import json
import warnings

warnings.filterwarnings('ignore')

class CryptoDataFetcher:
    HF_DATASET_ID = 'zongowo111/v2-crypto-ohlcv-data'
    HF_STAGE1_REPO = 'zongowo111/BB-Bounce-Validity-Classifier-Stage1'
    
    TIMEFRAMES = {
        '15m': 'BTC_15m.parquet',
        '1h': 'BTC_1h.parquet',
    }
    
    def __init__(self, cache_dir='./data/raw'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_symbol_timeframe(self, symbol, timeframe='15m'):
        filename = f'{symbol.replace("USDT", "")}_{timeframe}.parquet'
        file_path = self.cache_dir / 'datasets--zongowo111--v2-crypto-ohlcv-data' / 'klines' / symbol / filename
        
        if file_path.exists():
            df = pd.read_parquet(file_path)
            return df
        
        try:
            dataset = load_dataset(
                self.HF_DATASET_ID,
                data_files=f'klines/{symbol}/{filename}',
                cache_dir=str(self.cache_dir)
            )
            df = dataset['train'].to_pandas()
            return df
        except Exception as e:
            print(f'Error downloading {symbol} {timeframe}: {str(e)}')
            return None
    
    @staticmethod
    def download_stage1_classifier(model_type='lstm_validity', cache_dir='./models/stage1'):
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        files_to_download = {
            'lstm_validity': [
                'lstm_validity_model.h5',
                'lstm_validity_scaler.pkl',
                'lstm_validity_config.json'
            ],
            'gbm_validity': [
                'gbm_validity_model.pkl',
                'gbm_validity_scaler.pkl',
                'gbm_validity_config.json'
            ],
            'position_classifier': [
                'position_classifier_model.h5',
                'position_classifier_scaler.pkl',
                'position_classifier_config.json'
            ]
        }
        
        if model_type not in files_to_download:
            print(f'Available models: {list(files_to_download.keys())}')
            return None
        
        print(f'Downloading Stage 1 {model_type} classifier...')
        
        downloaded_files = {}
        for file_name in files_to_download[model_type]:
            try:
                local_path = hf_hub_download(
                    repo_id=CryptoDataFetcher.HF_STAGE1_REPO,
                    filename=file_name,
                    cache_dir=str(cache_dir),
                    force_download=False
                )
                downloaded_files[file_name] = local_path
                print(f'  Downloaded: {file_name}')
            except Exception as e:
                print(f'  Error downloading {file_name}: {str(e)}')
        
        return downloaded_files
    
    @staticmethod
    def load_stage1_model(model_type='lstm_validity', model_dir='./models/stage1'):
        import tensorflow as tf
        
        model_dir = Path(model_dir)
        model_file = model_dir / f'{model_type}_model.h5'
        scaler_file = model_dir / f'{model_type}_scaler.pkl'
        config_file = model_dir / f'{model_type}_config.json'
        
        if not model_file.exists():
            print(f'Model file not found: {model_file}')
            print('Attempting to download from Hugging Face...')
            CryptoDataFetcher.download_stage1_classifier(model_type, model_dir.parent)
        
        try:
            model = tf.keras.models.load_model(str(model_file))
            
            with open(scaler_file, 'rb') as f:
                scaler = pickle.load(f)
            
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            return model, scaler, config
        except Exception as e:
            print(f'Error loading model: {str(e)}')
            return None, None, None


if __name__ == '__main__':
    print('Fetching BTC_15m data...')
    fetcher = CryptoDataFetcher()
    btc_15m = fetcher.fetch_symbol_timeframe('BTCUSDT', '15m')
    
    if btc_15m is not None:
        print(f'BTC_15m shape: {btc_15m.shape}')
        print(f'Columns: {list(btc_15m.columns)}')
        print(f'Date range: {btc_15m["timestamp"].min()} to {btc_15m["timestamp"].max()}')
    
    print('\nDownloading Stage 1 Classifier...')
    files = CryptoDataFetcher.download_stage1_classifier('lstm_validity')
    if files:
        print(f'Successfully downloaded {len(files)} files')
