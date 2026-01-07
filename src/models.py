import numpy as np
import pandas as pd
from tensorflow import keras
from keras import layers, models
import lightgbm as lgb
from typing import Tuple, Dict
import json
import pickle
from pathlib import Path

class LSTMXGBoostModel:
    def __init__(self, timesteps=60, n_features=86, n_classes=5):
        self.timesteps = timesteps
        self.n_features = n_features
        self.n_classes = n_classes
        self.lstm_model = None
        self.gbm_model = None
        self.lstm_feature_dim = 32
    
    def build_lstm_encoder(self) -> models.Model:
        inputs = layers.Input(shape=(self.timesteps, self.n_features))
        
        x = layers.LSTM(128, activation='relu', return_sequences=True)(inputs)
        x = layers.Dropout(0.3)(x)
        x = layers.LSTM(64, activation='relu', return_sequences=False)(x)
        x = layers.Dropout(0.3)(x)
        
        return models.Model(inputs, x)
    
    def build_classifier(self) -> models.Model:
        inputs = layers.Input(shape=(self.timesteps, self.n_features))
        lstm_out = self.build_lstm_encoder()(inputs)
        
        x = layers.Dense(32, activation='relu')(lstm_out)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(16, activation='relu')(x)
        outputs = layers.Dense(self.n_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        return model
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_seq = []
        y_seq = []
        
        for i in range(len(X) - self.timesteps):
            X_seq.append(X[i:i + self.timesteps])
            y_seq.append(y[i + self.timesteps])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, early_stopping_patience=10) -> Dict:
        self.lstm_model = self.build_classifier()
        
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True
        )
        
        history = self.lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        
        return {'lstm_train_acc': train_acc, 'lstm_val_acc': val_acc}
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        predictions = self.lstm_model.predict(X_test, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        return {
            'accuracy': accuracy_score(y_test, predicted_classes),
            'precision': precision_score(y_test, predicted_classes, average='weighted', zero_division=0),
            'recall': recall_score(y_test, predicted_classes, average='weighted', zero_division=0),
            'f1': f1_score(y_test, predicted_classes, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, predicted_classes)
        }
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        predictions = self.lstm_model.predict(X, verbose=0)
        classes = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        return classes, confidences
    
    def save(self, model_dir: str):
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        self.lstm_model.save(model_dir / 'lstm_model.h5')
        
        config = {
            'timesteps': self.timesteps,
            'n_features': self.n_features,
            'n_classes': self.n_classes
        }
        with open(model_dir / 'config.json', 'w') as f:
            json.dump(config, f)
        
        print(f'Model saved to {model_dir}')
    
    def load(self, model_dir: str):
        model_dir = Path(model_dir)
        self.lstm_model = keras.models.load_model(model_dir / 'lstm_model.h5')
        
        with open(model_dir / 'config.json', 'r') as f:
            config = json.load(f)
        
        self.timesteps = config['timesteps']
        self.n_features = config['n_features']
        self.n_classes = config['n_classes']
        
        print(f'Model loaded from {model_dir}')


class VolatilityModel:
    def __init__(self, atr_multiplier=1.5, atr_window=14):
        self.atr_multiplier = atr_multiplier
        self.atr_window = atr_window
        self.model = None
    
    def create_labels(self, df: pd.DataFrame) -> np.ndarray:
        true_range = df[['high', 'low', 'open', 'close']].apply(
            lambda row: max(row['high'] - row['low'], abs(row['high'] - row['close']), abs(row['low'] - row['close'])),
            axis=1
        )
        atr = true_range.rolling(window=self.atr_window, min_periods=1).mean()
        atr_ma = atr.rolling(window=20, min_periods=1).mean()
        
        labels = (atr > atr_ma * self.atr_multiplier).astype(int)
        return labels.values
    
    def create_features(self, df: pd.DataFrame) -> np.ndarray:
        features = []
        
        true_range = df[['high', 'low', 'open', 'close']].apply(
            lambda row: max(row['high'] - row['low'], abs(row['high'] - row['close']), abs(row['low'] - row['close'])),
            axis=1
        ).values
        
        for period in [5, 10, 14, 20]:
            atr = pd.Series(true_range).rolling(window=period, min_periods=1).mean().values
            features.append(atr)
            features.append(atr / (df['close'].values + 1e-8))
        
        volatility = df['close'].rolling(window=20, min_periods=1).std().values
        features.append(volatility)
        features.append(volatility / (df['close'].rolling(window=20, min_periods=1).mean().values + 1e-8))
        
        if 'volume' in df.columns:
            features.append(df['volume'].values)
            vol_ma = df['volume'].rolling(window=20, min_periods=1).mean().values
            features.append(df['volume'].values / (vol_ma + 1e-8))
        
        return np.column_stack(features)
    
    def train(self, X_train, y_train, X_val, y_val) -> Dict:
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'verbose': -1
        }
        
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=100,
            callbacks=[lgb.early_stopping(10)]
        )
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        val_pred = (self.model.predict(X_val) > 0.5).astype(int)
        
        return {
            'accuracy': accuracy_score(y_val, val_pred),
            'precision': precision_score(y_val, val_pred, zero_division=0),
            'recall': recall_score(y_val, val_pred, zero_division=0),
            'f1': f1_score(y_val, val_pred, zero_division=0)
        }
