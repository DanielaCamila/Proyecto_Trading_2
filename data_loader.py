import pandas as pd
from config import DATA_PATH


def load_data():
    """
    Carga, limpia y valida los datos históricos de precios desde un archivo CSV.

    Este proceso incluye:
    - Carga de datos desde la ruta especificada en 'config.py'.
    - Estandarización de los nombres de las columnas (minúsculas, sin espacios).
    - Conversión de la columna de fecha a formato datetime, eliminando filas
      con fechas inválidas.
    - Establecimiento de la fecha como índice del DataFrame.
    - Detección y eliminación de registros duplicados basados en el índice.
    - Ordenamiento cronológico de los datos.
    - Verificación de la existencia de las columnas requeridas (OHLCV).
    - Selección final de las columnas necesarias para el análisis.

    Returns:
        pd.DataFrame: Un DataFrame limpio, validado y listo para el análisis.
                      Retorna None si ocurre un error durante la carga o el
                      procesamiento.
    """
    try:
        # --- 1. Carga inicial y estandarización ---
        df = pd.read_csv(DATA_PATH)
        # Se normalizan los nombres de columnas a minúsculas y sin espacios.
        df.columns = [col.strip().lower() for col in df.columns]

        # --- 2. Procesamiento de la columna de fecha ---
        # Se convierte la columna 'date' a formato datetime.
        # 'errors=coerce' transforma cualquier formato de fecha inválido en NaT (Not a Time).
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M', errors='coerce')

        # Se eliminan las filas que no pudieron ser convertidas a una fecha válida.
        initial_rows = len(df)
        df.dropna(subset=['date'], inplace=True)
        final_rows = len(df)

        if initial_rows > final_rows:
            rows_dropped = initial_rows - final_rows
            print(f"  - Información: Se eliminaron {rows_dropped} filas con formato de fecha inválido.")

        # --- 3. Indexación y limpieza de duplicados ---
        df.set_index('date', inplace=True)

        if df.index.has_duplicates:
            duplicates_count = df.index.duplicated().sum()
            print(f"  - Información: Se encontraron y eliminaron {duplicates_count} registros duplicados.")
            df = df[~df.index.duplicated(keep='first')]

        # Se ordena el índice para asegurar la cronología de la serie de tiempo.
        df.sort_index(inplace=True)

        # --- 4. Validación y selección de columnas ---
        # Se renombra 'volume usdt' a 'volume' por compatibilidad con librerías.
        if 'volume usdt' in df.columns:
            df.rename(columns={'volume usdt': 'volume'}, inplace=True)

        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise KeyError(f"Error de validación: Una o más columnas requeridas ({required_cols}) no se encontraron.")

        # Se retorna una copia del DataFrame solo con las columnas necesarias.
        final_df = df[required_cols].copy()

        return final_df

    except FileNotFoundError:
        print(f"  - Error Crítico: No se encontró el archivo de datos en la ruta especificada: {DATA_PATH}")
        return None
    except KeyError as e:
        print(f"  - Error Crítico: {e}")
        return None
    except Exception as e:
        print(f"  - Error Crítico: Ocurrió un error inesperado al procesar los datos: {e}")
        return None