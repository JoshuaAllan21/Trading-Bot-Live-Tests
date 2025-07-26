import yfinance as yf
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, BatchNormalization, GaussianNoise, Bidirectional, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle

# === Technical Indicators ===
class TechnicalIndicators:
    @staticmethod
    def compute_RSI(data, window=14):
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        RS = gain / loss
        data['RSI'] = 100 - (100 / (1 + RS))
        return data

    @staticmethod
    def compute_MACD(data, fast=12, slow=26, signal=9):
        data['EMA_fast'] = data['Close'].ewm(span=fast, adjust=False).mean()
        data['EMA_slow'] = data['Close'].ewm(span=slow, adjust=False).mean()
        data['MACD'] = data['EMA_fast'] - data['EMA_slow']
        data['MACD_signal'] = data['MACD'].ewm(span=signal, adjust=False).mean()
        data.drop(['EMA_fast', 'EMA_slow'], axis=1, inplace=True)
        return data

    @staticmethod
    def compute_BollingerBands(data, window=20, num_std=2):
        rolling_mean = data['Close'].rolling(window=window).mean()
        rolling_std = data['Close'].rolling(window=window).std()
        data['BB_upper'] = rolling_mean + (rolling_std * num_std)
        data['BB_lower'] = rolling_mean - (rolling_std * num_std)
        return data

# === Data Handler ===
class DataHandler:
    def __init__(self, stock_symbol):
        self.stock_symbol = stock_symbol
        self.daily_data = None
        self.hourly_data = None

    def download_daily_data(self, start, end):
        self.daily_data = yf.download(self.stock_symbol, start=start, end=end, interval="1d", auto_adjust=False)
        if self.daily_data.empty:
            raise ValueError(f"No daily data found for {self.stock_symbol}")
        self.daily_data = TechnicalIndicators.compute_RSI(self.daily_data.copy())
        self.daily_data = TechnicalIndicators.compute_MACD(self.daily_data.copy())
        self.daily_data = TechnicalIndicators.compute_BollingerBands(self.daily_data.copy())
        self.daily_data['MA7'] = self.daily_data['Close'].rolling(window=7).mean()
        self.daily_data['MA21'] = self.daily_data['Close'].rolling(window=21).mean()
        self.daily_data.dropna(inplace=True)

    def download_hourly_data(self, start, end):
        self.hourly_data = yf.download(self.stock_symbol, start=start, end=end, interval="1h", auto_adjust=False)
        if self.hourly_data.empty:
            raise ValueError(f"No hourly data found for {self.stock_symbol}")
        self.hourly_data['MA4'] = self.hourly_data['Close'].rolling(window=4).mean()
        self.hourly_data.dropna(inplace=True)

    def create_sequences_daily_classification(self, seq_length=21, pred_length=7):
        sequences = []
        targets = []
        data = self.daily_data.copy()
        for i in range(0, len(data) - seq_length - pred_length + 1):
            seq = data.iloc[i: i + seq_length]
            future = data.iloc[i + seq_length: i + seq_length + pred_length]
            sequences.append(seq.values)
            start_price = future.iloc[0]['Close']
            end_price = future.iloc[-1]['Close']
            trend_pct = float((end_price - start_price) / start_price) * 100
            if trend_pct > 0:
                targets.append(2)  # UP
            elif trend_pct < 0:
                targets.append(0)  # DOWN
            else:
                targets.append(1)  # FLAT
        return np.array(sequences), np.array(targets)

    def create_sequences_hourly(self, seq_length=168, pred_length=24):
        sequences = []
        targets = []
        data = self.hourly_data.copy()
        for i in range(0, len(data) - seq_length - pred_length + 1):
            seq = data.iloc[i: i + seq_length]
            future = data.iloc[i + seq_length: i + seq_length + pred_length]
            sequences.append(seq.values)
            start_price = data.iloc[i + seq_length - 1]['Close']
            future_low = future['Low'].min()
            low_pct_change = (future_low - start_price) / start_price
            targets.append(low_pct_change)
        return np.array(sequences), np.array(targets)

# === Model Builders ===
def build_trend_model(input_shape):
    model = Sequential()
    model.add(GaussianNoise(0.02, input_shape=input_shape))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(GlobalAveragePooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.0007), loss=CategoricalCrossentropy(), metrics=['accuracy'])
    return model

def build_low_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.3))
    model.add(LSTM(64))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.0005), loss=MeanSquaredError())
    return model

# === Train and Save Models ===
def train_and_save_models(stock_symbol, start_date, end_date):
    print(f"Preparing {stock_symbol}...")
    data_handler = DataHandler(stock_symbol)
    data_handler.download_daily_data(start_date, end_date)
    data_handler.download_hourly_data(start_date, end_date)

    X_daily, y_daily = data_handler.create_sequences_daily_classification(seq_length=21, pred_length=7)
    X_hourly, y_hourly = data_handler.create_sequences_hourly(seq_length=168, pred_length=24)

    print("Daily y class counts:", np.unique(y_daily, return_counts=True))
    print("Hourly y stats: min", np.min(y_hourly), "max", np.max(y_hourly))
    print("Sample X_daily shape:", X_daily.shape)
    print("Sample X_hourly shape:", X_hourly.shape)

    # Shuffle!
    X_daily, y_daily = shuffle(X_daily, y_daily, random_state=42)
    X_hourly, y_hourly = shuffle(X_hourly, y_hourly, random_state=42)

    y_daily_cat = to_categorical(y_daily, num_classes=3)

    model_trend = build_trend_model(X_daily.shape[1:])
    model_low = build_low_model(X_hourly.shape[1:])

    print("\n=== Trend Model Summary ===")
    print(model_trend.summary())
    print("\n=== Low Price Model Summary ===")
    print(model_low.summary())

    early_stopping_trend = EarlyStopping(monitor='loss', patience=4, restore_best_weights=True)
    reduce_lr_trend = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, min_lr=1e-6)

    early_stopping_low = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    reduce_lr_low = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, min_lr=1e-6)

    print(f"Training trend model...")
    history_trend = model_trend.fit(X_daily, y_daily_cat, epochs=30, batch_size=32, verbose=1, callbacks=[early_stopping_trend, reduce_lr_trend])

    print(f"Training low price model...")
    history_low = model_low.fit(X_hourly, y_hourly, epochs=15, batch_size=64, verbose=1, callbacks=[early_stopping_low, reduce_lr_low])

    # Save
    os.makedirs('models', exist_ok=True)
    model_trend.save("models/trend_model.keras")
    model_low.save("models/low_price_model.keras")
    print(f"Models saved to 'models/' folder.")

    # Plot losses
    plt.figure(figsize=(8,5))
    plt.plot(history_trend.history['loss'], label='Trend Model Loss')
    plt.title('Trend Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(8,5))
    plt.plot(history_low.history['loss'], label='Low Price Model Loss')
    plt.title('Low Price Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

    # Show a few sample predictions (trend and regression)
    print("\n=== Sample predictions on training data ===")
    print("Trend model predictions (first 10):")
    preds_trend = model_trend.predict(X_daily[:10])
    print(preds_trend)
    print("Trend model true labels (first 10):")
    print(y_daily[:10])
    print("Low price model predictions (first 10):")
    preds_low = model_low.predict(X_hourly[:10])
    print(preds_low.flatten())
    print("Low price model true targets (first 10):")
    print(y_hourly[:10])

# === Main Entry ===
if __name__ == "__main__":
    start_date = '2024-07-01'
    end_date = '2025-07-01'
    stock_list = ['AAPL']
    train_and_save_models(stock_list, start_date, end_date)

