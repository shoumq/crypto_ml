import requests
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import datetime
import pandas as pd
from fastapi import FastAPI, BackgroundTasks
import joblib
from pathlib import Path
import asyncio

MODEL_PATH = "crypto_model.pkl"

app = FastAPI()
is_training = False  # Флаг для отслеживания состояния обучения

def train_and_save_model(df):
    model = train_model(df)
    joblib.dump(model, MODEL_PATH)

def load_model():
    if Path(MODEL_PATH).exists():
        return joblib.load(MODEL_PATH)
    return None

async def train_model_async(crypto: str):
    global is_training
    if is_training:
        return
    is_training = True
    try:
        loop = asyncio.get_event_loop()
        # Загрузка данных в отдельном потоке
        btc_data = await loop.run_in_executor(None, get_crypto_data, crypto)
        # Подготовка данных
        prepared_data, _ = await loop.run_in_executor(None, prepare_features, btc_data)
        # Обучение модели
        await loop.run_in_executor(None, train_and_save_model, prepared_data)
    finally:
        is_training = False

@app.on_event("startup")
async def startup_event():
    # Асинхронно запускаем обучение модели при старте
    asyncio.create_task(train_model_async("BTC"))

def get_crypto_data(crypto):
    end_time = datetime.datetime.now(datetime.timezone.utc)
    rounded_end_time = end_time.replace(minute=0, second=0, microsecond=0)
    if rounded_end_time > end_time:
        rounded_end_time -= datetime.timedelta(hours=1)
    
    end_timestamp = int(rounded_end_time.timestamp() * 1000)
    
    url = "https://api.bybit.com/v5/market/kline"
    limit = 1000
    all_data = []
    
    current_end = end_timestamp
    max_iterations = 50  # Защита от бесконечного цикла
    while len(all_data) < 50000 and max_iterations > 0:
        params = {
            "category": "linear",
            "symbol": f"{crypto}USDT",
            "interval": "60",
            "end": current_end,
            "limit": limit
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        if data["retCode"] != 0:
            raise Exception(f"API Error: {data['retMsg']}")
        
        klines = data["result"]["list"]
        if not klines:
            break
        
        all_data.extend(klines)
        # Обновляем current_end для следующего запроса
        current_end = int(float(klines[-1][0])) - 3600 * 1000  # Предыдущий час
        max_iterations -= 1
    
    all_data = all_data[:50000]
    
    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
    
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    
    df = df.sort_values('timestamp').set_index('timestamp')
    df = df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    return df

def prepare_features(df):
    df = df.copy()  # Создаем копию для устранения предупреждения
    df['SMA_24'] = df['Close'].rolling(24).mean()
    df['SMA_168'] = df['Close'].rolling(168).mean()
    df['SMA_720'] = df['Close'].rolling(720).mean()
    
    df['Volatility_24'] = df['Close'].rolling(24).std()
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    
    df['Bollinger_Upper'] = sma20 + 2 * std20
    df['Bollinger_Lower'] = sma20 - 2 * std20
    
    df = df.dropna()
    
    df['Target'] = df['Close'].shift(-1)
    
    last_row = df[df['Target'].isna()].copy()
    df = df.dropna()
    
    return df, last_row

def train_model(df):
    features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_24', 'SMA_168', 'SMA_720',
        'Volatility_24', 'RSI_14',
        'MACD', 'MACD_Signal',
        'Bollinger_Upper', 'Bollinger_Lower'
    ]
    
    X = df[features]
    y = df['Target']
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    param_grid = {
        'n_estimators': [200],
        'max_depth': [20],
        'min_samples_split': [10]
    }
    
    grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=tscv, verbose=1, n_jobs=-1)
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    
    last_train_idx, last_test_idx = list(tscv.split(X))[-1]
    X_train, X_test = X.iloc[last_train_idx], X.iloc[last_test_idx]
    y_train, y_test = y.iloc[last_train_idx], y.iloc[last_test_idx]
    
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"Лучшие параметры: {grid_search.best_params_}")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    
    return best_model

@app.get("/api/crypto/predict/{crypto}")
async def getPrediction(crypto: str, background_tasks: BackgroundTasks):
    global is_training
    try:
        model = load_model()
        if model is None:
            if not is_training:
                background_tasks.add_task(train_model_async, crypto)
            return {"message": "Модель обучается, попробуйте позже"}
        
        loop = asyncio.get_event_loop()
        # Загрузка данных в отдельном потоке
        btc_data = await loop.run_in_executor(None, get_crypto_data, crypto)
        prepared_data, last_row = await loop.run_in_executor(None, prepare_features, btc_data)
        
        if last_row.empty:
            raise ValueError("Нет данных для прогноза")
        
        features = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_24', 'SMA_168', 'SMA_720',
            'Volatility_24', 'RSI_14',
            'MACD', 'MACD_Signal',
            'Bollinger_Upper', 'Bollinger_Lower'
        ]
        prediction = model.predict(last_row[features])[0]
        
        current_price = last_row['Close'].iloc[0]
        next_time = last_row.index[0] + datetime.timedelta(hours=1)
        
        response = {
            "prediction": prediction, 
            "prediction_time": next_time.strftime('%Y-%m-%d %H:%M'),
            "current_price": current_price
        }
        
        return response
        
    except Exception as e:
        print(f"Ошибка: {e}")
        return {"error": str(e)}