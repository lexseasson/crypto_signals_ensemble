import pandas as pd
import numpy as np

def add_moving_averages(data: pd.DataFrame) -> pd.DataFrame:
    """Calcula las Medias Móviles Simples (SMA) de 20 y 50 periodos."""
    data['SMA_20'] = data['close'].rolling(window=20).mean()
    data['SMA_50'] = data['close'].rolling(window=50).mean()
    return data

def add_bollinger_bands(data: pd.DataFrame) -> pd.DataFrame:
    """Calcula las Bandas de Bollinger (BB)."""
    data['BB_middle'] = data['close'].rolling(window=20).mean()
    data['BB_std'] = data['close'].rolling(window=20).std()
    data['BB_upper'] = data['BB_middle'] + 2 * data['BB_std']
    data['BB_lower'] = data['BB_middle'] - 2 * data['BB_std']
    return data

def add_macd(data: pd.DataFrame) -> pd.DataFrame:
    """Calcula la Convergencia/Divergencia de Medias Móviles (MACD)."""
    data['EMA_12'] = data['close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    return data

def add_rsi(data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Calcula el Índice de Fuerza Relativa (RSI)."""
    delta = data['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Usar el método de Wilder para el promedio móvil (ewm)
    avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

def add_atr(data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Calcula el Rango Verdadero Promedio (ATR)."""
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift(1))
    low_close = np.abs(data['low'] - data['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR'] = tr.ewm(span=window, adjust=False).mean() # Usar EMA para suavizado
    return data

def add_stochastic(data: pd.DataFrame) -> pd.DataFrame:
    """Calcula el Oscilador Estocástico (%K y %D)."""
    data['14-high'] = data['high'].rolling(window=14).max()
    data['14-low'] = data['low'].rolling(window=14).min()
    data['%K'] = 100 * (data['close'] - data['14-low']) / (data['14-high'] - data['14-low'])
    data['%D'] = data['%K'].rolling(window=3).mean()
    return data

def add_ichimoku_cloud(data: pd.DataFrame) -> pd.DataFrame:
    """Calcula los componentes de la Nube de Ichimoku."""
    high9 = data['high'].rolling(window=9).max()
    low9 = data['low'].rolling(window=9).min()
    data['tenkan_sen'] = (high9 + low9) / 2

    high26 = data['high'].rolling(window=26).max()
    low26 = data['low'].rolling(window=26).min()
    data['kijun_sen'] = (high26 + low26) / 2

    data['senkou_span_a'] = ((data['tenkan_sen'] + data['kijun_sen']) / 2).shift(26)
    high52 = data['high'].rolling(window=52).max()
    low52 = data['low'].rolling(window=52).min()
    data['senkou_span_b'] = ((high52 + low52) / 2).shift(26)
    data['chikou_span'] = data['close'].shift(-26)
    
    return data

def add_supertrend(data: pd.DataFrame, atr_multiplier: float = 3, atr_window: int = 14) -> pd.DataFrame:
    """Calcula el indicador Supertrend."""
    data = add_atr(data, window=atr_window)
    hl2 = (data['high'] + data['low']) / 2
    
    # Bandas base
    data['Upper_Band'] = hl2 + (atr_multiplier * data['ATR'])
    data['Lower_Band'] = hl2 - (atr_multiplier * data['ATR'])
    
    # Lógica de Supertrend (simplificada para modularidad)
    data['Supertrend'] = np.nan
    data['Supertrend_Direction'] = np.nan
    
    for i in range(1, len(data)):
        # Lógica de seguimiento de tendencia
        if data['close'].iloc[i] > data['Upper_Band'].iloc[i-1]:
            data['Supertrend'].iloc[i] = data['Lower_Band'].iloc[i]
            data['Supertrend_Direction'].iloc[i] = 1 # Alcista
        elif data['close'].iloc[i] < data['Lower_Band'].iloc[i-1]:
            data['Supertrend'].iloc[i] = data['Upper_Band'].iloc[i]
            data['Supertrend_Direction'].iloc[i] = -1 # Bajista
        else:
            data['Supertrend'].iloc[i] = data['Supertrend'].iloc[i-1]
            data['Supertrend_Direction'].iloc[i] = data['Supertrend_Direction'].iloc[i-1]
            
            # Ajuste de bandas para evitar falsas señales
            if data['Supertrend_Direction'].iloc[i] == 1 and data['Lower_Band'].iloc[i] < data['Supertrend'].iloc[i-1]:
                data['Supertrend'].iloc[i] = data['Supertrend'].iloc[i-1]
            elif data['Supertrend_Direction'].iloc[i] == -1 and data['Upper_Band'].iloc[i] > data['Supertrend'].iloc[i-1]:
                data['Supertrend'].iloc[i] = data['Supertrend'].iloc[i-1]
                
    return data.drop(columns=['Upper_Band', 'Lower_Band'], errors='ignore') # Limpiar columnas intermedias

def add_vwap(data: pd.DataFrame) -> pd.DataFrame:
    """Calcula el Precio Promedio Ponderado por Volumen (VWAP)."""
    q = data['volume']
    p = data['close']
    data['VWAP'] = (p * q).cumsum() / q.cumsum()
    return data

def add_rvol(data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Calcula el Volumen Relativo (RVOL)."""
    data['volume_avg'] = data['volume'].rolling(window=window).mean()
    data['RVOL'] = data['volume'] / data['volume_avg']
    return data.drop(columns=['volume_avg'], errors='ignore')

def calculate_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Aplica todos los indicadores técnicos al DataFrame."""
    data = add_moving_averages(data)
    data = add_bollinger_bands(data)
    data = add_macd(data)
    data = add_rsi(data)
    data = add_atr(data)
    data = add_stochastic(data)
    data = add_ichimoku_cloud(data)
    data = add_supertrend(data)
    data = add_vwap(data)
    data = add_rvol(data)
    return data.dropna() # Eliminar filas con NaN generadas por los cálculos iniciales
