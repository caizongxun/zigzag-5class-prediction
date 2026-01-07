import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

class ZigZagIndicator:
    LABEL_MAP = {
        0: 'NO_SIGNAL',
        1: 'HH',
        2: 'LH',
        3: 'HL',
        4: 'LL'
    }
    
    LABEL_NAMES = {
        'NO_SIGNAL': 0,
        'HH': 1,
        'LH': 2,
        'HL': 3,
        'LL': 4
    }
    
    def __init__(self, depth=12, deviation=5, backstep=2):
        self.depth = depth
        self.deviation = deviation
        self.backstep = backstep
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        high = df['high'].values
        low = df['low'].values
        
        zigzag = np.zeros(len(df))
        pivot_type = 0
        last_pivot_idx = 0
        last_pivot_value = high[0]
        
        for i in range(self.depth, len(df)):
            if pivot_type == 0:
                if high[i] > last_pivot_value:
                    last_pivot_value = high[i]
                    last_pivot_idx = i
                else:
                    pct_change = abs((low[i] - last_pivot_value) / last_pivot_value) * 100
                    if pct_change >= self.deviation:
                        zigzag[last_pivot_idx] = last_pivot_value
                        pivot_type = -1
                        last_pivot_value = low[i]
                        last_pivot_idx = i
            elif pivot_type == -1:
                if low[i] < last_pivot_value:
                    last_pivot_value = low[i]
                    last_pivot_idx = i
                else:
                    pct_change = abs((high[i] - last_pivot_value) / last_pivot_value) * 100
                    if pct_change >= self.deviation:
                        zigzag[last_pivot_idx] = last_pivot_value
                        pivot_type = 1
                        last_pivot_value = high[i]
                        last_pivot_idx = i
        
        return pd.Series(zigzag, index=df.index)
    
    def label_kbars(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['zigzag_value'] = self.calculate(df)
        
        labels = np.zeros(len(df), dtype=int)
        
        if len(df) < 2:
            df['zigzag_label'] = labels
            return df
        
        high = df['high'].values
        low = df['low'].values
        zigzag = df['zigzag_value'].values
        
        pivots = []
        for i in range(len(zigzag)):
            if zigzag[i] != 0:
                pivots.append((i, zigzag[i]))
        
        if len(pivots) >= 2:
            for i in range(1, len(pivots)):
                prev_idx, prev_value = pivots[i - 1]
                curr_idx, curr_value = pivots[i]
                
                if prev_value == low[prev_idx]:
                    if curr_value > prev_value:
                        for j in range(prev_idx, curr_idx + 1):
                            if high[j] > prev_value and labels[j] == 0:
                                labels[j] = self.LABEL_NAMES['HL']
                    else:
                        for j in range(prev_idx, curr_idx + 1):
                            if high[j] < prev_value and labels[j] == 0:
                                labels[j] = self.LABEL_NAMES['LH']
                
                elif prev_value == high[prev_idx]:
                    if curr_value < prev_value:
                        for j in range(prev_idx, curr_idx + 1):
                            if low[j] < prev_value and labels[j] == 0:
                                labels[j] = self.LABEL_NAMES['LL']
                    else:
                        for j in range(prev_idx, curr_idx + 1):
                            if high[j] > prev_value and labels[j] == 0:
                                labels[j] = self.LABEL_NAMES['HH']
        
        df['zigzag_label'] = labels
        return df
    
    @staticmethod
    def get_label_name(label_id: int) -> str:
        return ZigZagIndicator.LABEL_MAP.get(label_id, 'UNKNOWN')
    
    @staticmethod
    def get_label_id(label_name: str) -> int:
        return ZigZagIndicator.LABEL_NAMES.get(label_name, -1)
