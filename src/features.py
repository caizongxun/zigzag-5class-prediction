import pandas as pd
import numpy as np
from typing import List, Tuple

class FeatureEngineer:
    def __init__(self, lookback_periods=None):
        self.lookback_periods = lookback_periods or [5, 10, 20, 50, 200]
    
    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df = self._add_basic_features(df)
        df = self._add_moving_average_features(df)
        df = self._add_momentum_features(df)
        df = self._add_volatility_features(df)
        df = self._add_volume_features(df)
        df = self._add_structure_features(df)
        df = self._add_advanced_indicators(df)
        
        return df
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['high_low_range'] = df['high'] - df['low']
        df['open_close_range'] = df['close'] - df['open']
        df['high_close_range'] = df['high'] - df['close']
        df['low_close_range'] = df['close'] - df['low']
        df['true_range'] = df[['high_low_range', 'high_close_range', 'low_close_range']].max(axis=1)
        
        for period in [1, 2, 5, 10]:
            df[f'price_change_{period}'] = df['close'].diff(period)
            df[f'price_pct_change_{period}'] = df['close'].pct_change(period)
            df[f'high_change_{period}'] = df['high'].diff(period)
            df[f'low_change_{period}'] = df['low'].diff(period)
        
        return df
    
    def _add_moving_average_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for period in self.lookback_periods:
            if period <= len(df):
                df[f'sma_{period}'] = df['close'].rolling(window=period, min_periods=1).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
                
                if f'sma_{period}' in df.columns and f'sma_20' in df.columns:
                    df[f'price_ma_ratio_{period}'] = df['close'] / (df[f'sma_{period}'] + 1e-8)
        
        df['trend_strength_20'] = (df['sma_20'] - df['sma_50']) / (df['close'] + 1e-8)
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for period in [5, 10, 14, 20]:
            if period <= len(df):
                deltas = df['close'].diff()
                gains = (deltas.where(deltas > 0, 0)).rolling(window=period, min_periods=1).mean()
                losses = (-deltas.where(deltas < 0, 0)).rolling(window=period, min_periods=1).mean()
                rs = gains / (losses + 1e-8)
                df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        for period in [12, 26]:
            if period <= len(df):
                df[f'macd_line_{period}'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        
        for period in [5, 10, 20]:
            if period <= len(df):
                df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for period in [5, 10, 20]:
            if period <= len(df):
                df[f'volatility_{period}'] = df['close'].rolling(window=period, min_periods=1).std()
                
                atr = df['true_range'].rolling(window=period, min_periods=1).mean()
                df[f'atr_{period}'] = atr
                df[f'atr_pct_{period}'] = (atr / df['close']) * 100
        
        df['volatility_ratio_20_5'] = (df['volatility_20'] / (df['volatility_5'] + 1e-8))
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'volume' not in df.columns:
            df['volume'] = 1
        
        for period in [5, 10, 20]:
            if period <= len(df):
                df[f'volume_ma_{period}'] = df['volume'].rolling(window=period, min_periods=1).mean()
                df[f'volume_ratio_{period}'] = df['volume'] / (df[f'volume_ma_{period}'] + 1e-8)
        
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        return df
    
    def _add_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['highest_high_20'] = df['high'].rolling(window=20, min_periods=1).max()
        df['lowest_low_20'] = df['low'].rolling(window=20, min_periods=1).min()
        df['highest_high_50'] = df['high'].rolling(window=50, min_periods=1).max()
        df['lowest_low_50'] = df['low'].rolling(window=50, min_periods=1).min()
        
        df['distance_resistance_20'] = (df['highest_high_20'] - df['close']) / (df['close'] + 1e-8)
        df['distance_support_20'] = (df['close'] - df['lowest_low_20']) / (df['close'] + 1e-8)
        df['distance_resistance_50'] = (df['highest_high_50'] - df['close']) / (df['close'] + 1e-8)
        df['distance_support_50'] = (df['close'] - df['lowest_low_50']) / (df['close'] + 1e-8)
        
        up_down = pd.Series(np.where(df['close'] > df['open'], 1, -1), index=df.index)
        df['consecutive_bars'] = up_down * (up_down.groupby((up_down != up_down.shift()).cumsum()).cumcount() + 1)
        
        return df
    
    def _add_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        for period in [10, 20]:
            if period <= len(df):
                mid_line = (df['highest_high_20'] + df['lowest_low_20']) / 2
                df[f'bb_upper_{period}'] = mid_line + (df['volatility_20'] * 2)
                df[f'bb_lower_{period}'] = mid_line - (df['volatility_20'] * 2)
                df[f'bb_width_{period}'] = df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']
                df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_width_{period}'] + 1e-8)
        
        df['volatility_regime'] = df['atr_5'] / (df['atr_20'] + 1e-8)
        df['price_position_20'] = (df['close'] - df['lowest_low_20']) / (df['highest_high_20'] - df['lowest_low_20'] + 1e-8)
        
        return df
    
    @staticmethod
    def get_feature_columns(df: pd.DataFrame) -> List[str]:
        exclude_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'zigzag_value', 'zigzag_label']
        feature_cols = [col for col in df.columns if col not in exclude_cols and col in df.columns]
        return [col for col in feature_cols if df[col].dtype in ['float64', 'float32', 'int64', 'int32']]
