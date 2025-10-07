"""
Script principal para la ejecución del backtesting de la estrategia de trading.

Este script orquesta el proceso completo, que incluye:
1. Carga y preprocesamiento de datos históricos.
2. División de los datos en conjuntos de entrenamiento, validación y prueba.
3. Optimización de hiperparámetros en el conjunto de entrenamiento.
4. Validación del modelo con los parámetros óptimos en el conjunto de validación.
5. Evaluación final del rendimiento en el conjunto de prueba (datos no vistos).
"""

import pandas as pd
from optimizer import Optimizer
from backtester import Backtester
from data_loader import load_data
from indicator_calculator import add_indicators
from reporting import plot_portfolio_value, generate_performance_report
from config import INITIAL_CASH


def print_results(phase_name, portfolio_series, initial_cash):
    """
    Imprime en consola un resumen estandarizado del rendimiento de una fase
    del backtesting.

    Args:
        phase_name (str): El nombre de la fase (ej. "Entrenamiento", "Validación").
        portfolio_series (pd.Series): Serie de tiempo con el valor del portafolio.
        initial_cash (float): El capital inicial para el cálculo de retornos.
    """
    print(f"\n------ Resumen de Resultados: Fase de {phase_name} ------")

    if portfolio_series.empty or portfolio_series.iloc[-1] <= 0:
        print("Análisis: No se generaron operaciones o el capital final fue nulo/negativo.")
        return

    final_value = portfolio_series.iloc[-1]
    total_return_pct = (final_value - initial_cash) / initial_cash * 100

    # Cálculo del retorno anualizado (APR) basado en la frecuencia horaria de los datos.
    n_hours = (portfolio_series.index[-1] - portfolio_series.index[0]).total_seconds() / 3600.0
    annualized_return_pct = 0.0
    if n_hours > 0:
        # Se calcula la tasa de retorno por hora y se capitaliza a un año (8760 horas).
        annualized_return = (1 + (total_return_pct / 100)) ** (8760.0 / n_hours) - 1
        annualized_return_pct = annualized_return * 100

    print(f"Capital Inicial: ${initial_cash:,.2f}")
    print(f"Capital Final:   ${final_value:,.2f}")
    print(f"Retorno Total:   {total_return_pct:.2f}%")
    print(f"Retorno Anualizado (APR): {annualized_return_pct:.2f}%")
    print("--------------------------------------------------")


def main():
    """
    Función principal que ejecuta el flujo completo del backtesting.
    """
    print("==========================================================")
    print("INICIO DEL PROCESO DE BACKTESTING Y OPTIMIZACIÓN")
    print("División de datos: 60% Entrenamiento, 20% Validación, 20% Prueba")
    print("==========================================================")

    # 1. Carga y División de Datos
    print("\n[Paso 1/4] Cargando y dividiendo los datos...")
    market_data = load_data()
    if market_data is None:
        print("Error: No se pudieron cargar los datos. Abortando proceso.")
        return

    train_end_index = int(len(market_data) * 0.6)
    validation_end_index = int(len(market_data) * 0.8)

    train_set = market_data.iloc[0:train_end_index]
    validation_set = market_data.iloc[train_end_index:validation_end_index]
    test_set = market_data.iloc[validation_end_index:]

    print(f"  - Tamaño del set de Entrenamiento: {len(train_set)} registros.")
    print(f"  - Tamaño del set de Validación:    {len(validation_set)} registros.")
    print(f"  - Tamaño del set de Prueba:      {len(test_set)} registros.")
    print("Datos cargados y divididos correctamente.")

    # 2. Fase de Optimización sobre el conjunto de entrenamiento
    print("\n[Paso 2/4] Ejecutando fase de optimización de parámetros...")
    optimizer = Optimizer(train_set)
    champion_params = optimizer.run_optimization(n_trials=50)  # Se puede ajustar n_trials

    if not champion_params:
        print("Error: La optimización no produjo parámetros válidos. Abortando proceso.")
        return

    # Se realiza un backtest sobre el conjunto de entrenamiento para tener una línea base.
    indicator_params = {
        'ema_len': champion_params['ema_len'], 'macd_fast': champion_params['macd_fast'],
        'macd_slow': champion_params['macd_slow'], 'macd_signal': champion_params['macd_signal'],
        'adx_len': champion_params['adx_len']
    }
    train_data_indicators = add_indicators(train_set.copy(), **indicator_params)
    train_backtester = Backtester(train_data_indicators, champion_params)
    train_portfolio = train_backtester.run(return_value_series=True)
    print_results("Entrenamiento (Línea Base)", train_portfolio, INITIAL_CASH)
    if not train_portfolio.empty:
        generate_performance_report(train_portfolio, "train_performance.html")

    # 3. Fase de Validación con los parámetros óptimos
    print("\n[Paso 3/4] Ejecutando fase de validación (Out-of-Sample)...")
    validation_data_indicators = add_indicators(validation_set.copy(), **indicator_params)
    val_backtester = Backtester(validation_data_indicators, champion_params)
    val_portfolio = val_backtester.run(return_value_series=True)
    print_results("Validación", val_portfolio, INITIAL_CASH)
    if not val_portfolio.empty:
        generate_performance_report(val_portfolio, "validation_performance.html")
        plot_portfolio_value(val_portfolio, "validation_equity_curve.png")

    # 4. Fase de Prueba Final
    print("\n[Paso 4/4] Ejecutando prueba final en datos no vistos (Forward Testing)...")
    test_data_indicators = add_indicators(test_set.copy(), **indicator_params)
    test_backtester = Backtester(test_data_indicators, champion_params)
    test_portfolio = test_backtester.run(return_value_series=True)
    print_results("Prueba Final", test_portfolio, INITIAL_CASH)
    if not test_portfolio.empty:
        generate_performance_report(test_portfolio, "final_test_performance.html")
        plot_portfolio_value(test_portfolio, "final_test_equity_curve.png")

    print("\n==========================================================")
    print("PROCESO DE BACKTESTING FINALIZADO")
    print("==========================================================")


if __name__ == "__main__":
    main()