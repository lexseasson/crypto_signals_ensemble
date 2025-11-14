import pandas as pd
import ccxt
import pytz
from cachetools import TTLCache
import logging
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up caching with a TTL of 60 seconds for real-time data
cache = TTLCache(maxsize=100, ttl=60)

# Configuración de la API de Binance (usando variables de entorno para seguridad)
try:
    API_KEY = os.getenv('BINANCE_API_KEY')
    SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')
    
    if not API_KEY or not SECRET_KEY:
        logging.warning("Las claves API de Binance no están configuradas en el archivo .env. Se usará la conexión pública.")
        exchange = ccxt.binance()
    else:
        exchange = ccxt.binance({
            'apiKey': API_KEY,
            'secret': SECRET_KEY,
        })
        # Sincronizar la diferencia de tiempo con Binance
        exchange.load_time_difference() 

except Exception as e:
    logging.error(f"Error al inicializar CCXT: {e}")
    exchange = ccxt.binance() # Fallback a conexión pública

# Zona horaria de Argentina (o la que se prefiera)
# Se mantiene la zona horaria original del código para consistencia.
ARGENTINA_TZ = pytz.timezone('America/Argentina/Buenos_Aires')

def fetch_ohlcv_data(symbol: str, timeframe: str = '1m', limit: int = 1000) -> pd.DataFrame:
    """
    Obtiene datos OHLCV de un símbolo y los almacena en caché.

    Args:
        symbol (str): Par de criptomoneda (e.g., 'BTC/USDT').
        timeframe (str): Intervalo de tiempo (e.g., '1m', '1h').
        limit (int): Número máximo de velas a obtener.

    Returns:
        pd.DataFrame: DataFrame con los datos OHLCV.
    """
    cache_key = f"{symbol}_{timeframe}_{limit}"
    if cache_key in cache:
        logging.info(f"Datos de {symbol} ({timeframe}) obtenidos de la caché.")
        return cache[cache_key]
    
    try:
        logging.info(f"Obteniendo datos de {symbol} ({timeframe}) de la API...")
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Conversión y localización de la marca de tiempo
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(ARGENTINA_TZ)
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='first')]  # Eliminar duplicados en el índice
        df = df.sort_index()  # Asegurar que el índice esté ordenado
        
        cache[cache_key] = df
        logging.info(f"Datos de {symbol} ({timeframe}) almacenados en caché.")
        return df
    
    except Exception as e:
        logging.error(f"Error al obtener datos para {symbol}: {e}")
        return pd.DataFrame()

# Lista de pares de criptomonedas (se puede mover a un archivo de configuración si crece)
CRYPTO_PAIRS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'FTM/USDT', 'ARB/USDT',
    'AVAX/USDT', 'GMX/USDT', 'MATIC/USDT', 'DOT/USDT', 'AAVE/USDT', 'TIA/USDT',
    'LTC/USDT', 'AXS/USDT', 'OP/USDT', 'UNI/USDT', 'PENDLE/USDT', 'KAS/USDT', 'ALGO/USDT', 'XRP/USDT'
]
