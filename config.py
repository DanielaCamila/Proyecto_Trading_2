"""
Archivo de configuración central para el proyecto de backtesting.

Este script contiene todas las variables globales y parámetros que pueden
ser ajustados para modificar el comportamiento de la simulación sin tener
que alterar el código fuente de los módulos principales.
"""

# --- Parámetros de Datos ---
# Ruta relativa al archivo CSV que contiene los datos históricos de mercado.
DATA_PATH = 'Binance_BTCUSDT_1h.csv'

# --- Parámetros de la Simulación de Backtesting ---
# Capital inicial (en USD) para todas las simulaciones.
INITIAL_CASH = 1_000_000

# Comisión por operación (porcentaje). Representa el costo de transacción
# del broker. Ejemplo: 0.00125 equivale a 0.125%.
COMMISSION = 0.00125