import requests
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import datetime
import pandas as pd

def get_btc_data():
    end_time = datetime.datetime.now(datetime.timezone.utc)
    # Округляем до ближайшего завершенного часа
    rounded_end_time = end_time.replace(minute=0, second=0, microsecond=0)
    if rounded_end_time > end_time:
        rounded_end_time -= datetime.timedelta(hours=1)
    
    end_timestamp = int(rounded_end_time.timestamp() * 1000)
    
    url = "https://api.bybit.com/v5/market/kline"
    limit = 1000
    all_data = []
    
    current_end = end_timestamp
    while len(all_data) < 50000:
        params = {
            "category": "linear",
            "symbol": "BTCUSDT",
            "interval": "60",  # Часовой интервал
            "end": current_end,
            "limit": limit
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        if data["retCode"] != 0:
            raise Exception(f"API Error: {data['retMsg']}")
        
        klines = data["result"]["list"]
        all_data.extend(klines)
        
        if len(klines) < limit:
            break
        
        current_end = int(float(klines[-1][0]))
    
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
    # Скользящие средние
    df['SMA_24'] = df['Close'].rolling(24).mean()    # 24 часа
    df['SMA_168'] = df['Close'].rolling(168).mean()  # 1 неделя
    df['SMA_720'] = df['Close'].rolling(720).mean()  # 30 дней
    
    # Волатильность
    df['Volatility_24'] = df['Close'].rolling(24).std()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['Bollinger_Upper'] = sma20 + 2 * std20
    df['Bollinger_Lower'] = sma20 - 2 * std20
    
    # Удаление NaN
    df = df.dropna()
    
    # Целевая переменная (цена через час)
    df['Target'] = df['Close'].shift(-1)
    
    # Последняя строка для предсказания
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
    
    # Используем TimeSeriesSplit для временных рядов
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
    
    # Оценка на последнем фолде
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

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

def main():
    try:
        btc_data = get_btc_data()  # 1 год данных
        prepared_data, last_row = prepare_features(btc_data)
        
        if len(prepared_data) < 100:
            raise ValueError("Недостаточно данных для обучения")
            
        if last_row.empty:
            raise ValueError("Нет данных для прогноза")
        
        model = train_model(prepared_data)
        
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
        
        print(f"\nТекущая цена BTC: ${current_price:.2f}")
        print(f"Прогноз на {next_time.strftime('%Y-%m-%d %H:%M')}: ${prediction:.2f}")
        
        evaluate_model(model, prepared_data[features], prepared_data['Target'])
        
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()