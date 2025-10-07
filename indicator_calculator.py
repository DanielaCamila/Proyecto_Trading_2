import pandas as pd
import pandas_ta as ta

def add_indicators(df, ema_len, macd_fast, macd_slow, macd_signal, adx_len):
    """
    Calcula y añade un conjunto de indicadores técnicos a un DataFrame de datos de mercado.

    Utiliza la librería pandas-ta para los cálculos. Los indicadores añadidos son:
    - Media Móvil Exponencial (EMA) de largo plazo.
    - Convergencia/Divergencia de Medias Móviles (MACD).
    - Índice Direccional Promedio (ADX).

    Al final del proceso, se eliminan las filas iniciales que contienen valores
    NaN debido al período de cálculo de los indicadores.

    Args:
        df (pd.DataFrame): El DataFrame original con datos OHLCV.
        ema_len (int): El período para la EMA de largo plazo.
        macd_fast (int): El período de la EMA rápida para el MACD.
        macd_slow (int): El período de la EMA lenta para el MACD.
        macd_signal (int): El período de la línea de señal para el MACD.
        adx_len (int): El período para el cálculo del ADX.

    Returns:
        pd.DataFrame: Un nuevo DataFrame con las columnas de los indicadores añadidas.
    """
    # Se trabaja sobre una copia para evitar la advertencia 'SettingWithCopyWarning'.
    data = df.copy()

    # --- 1. Cálculo de Indicadores ---
    # Se utiliza el metodo de extensión .ta de pandas-ta para añadir los indicadores.
    # El argumento 'append=True' modifica el DataFrame 'data' directamente.

    # Media Móvil Exponencial (EMA)
    data.ta.ema(length=ema_len, append=True)

    # Convergencia/Divergencia de Medias Móviles (MACD)
    data.ta.macd(fast=macd_fast, slow=macd_slow, signal=macd_signal, append=True)

    # Índice Direccional Promedio (ADX)
    data.ta.adx(length=adx_len, append=True)

    # --- 2. Limpieza de Datos ---
    # Los indicadores técnicos generan valores NaN en las primeras filas
    # del DataFrame, ya que necesitan un histórico mínimo para ser calculados.
    # Estas filas se eliminan para asegurar la integridad de los datos en el backtest.
    data.dropna(inplace=True)

    return data