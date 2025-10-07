import optuna
import pandas as pd
from backtester import Backtester
from indicator_calculator import add_indicators


class Optimizer:
    """
    Clase encargada de encontrar la combinación óptima de parámetros para la
    estrategia de trading mediante la librería Optuna, utilizando una validación
    cruzada de tipo Walk-Forward.
    """

    def __init__(self, data):
        """
        Inicializa el optimizador.

        Args:
            data (pd.DataFrame): El conjunto de datos de mercado para la optimización.
        """
        self.data = data

    def _calculate_objective_metric(self, portfolio_value_series):
        """
        Calcula la métrica objetivo para la optimización (Calmar Ratio).

        El Calmar Ratio se define como el retorno anualizado dividido por el
        máximo drawdown. Este metodo está adaptado para datos con frecuencia horaria.

        Args:
            portfolio_value_series (pd.Series): Serie de tiempo con el valor del
                                               portafolio a lo largo del backtest.

        Returns:
            float: El valor calculado del Calmar Ratio. Retorna 0.0 si el cálculo
                   no es posible o si el drawdown es insignificante.
        """
        if portfolio_value_series.empty or len(portfolio_value_series) < 2:
            return 0.0

        total_return = (portfolio_value_series.iloc[-1] / portfolio_value_series.iloc[0]) - 1

        # Anualización basada en datos horarios (asumiendo 24/7).
        # Total de horas en un año = 365 * 24 = 8760.
        n_hours = (portfolio_value_series.index[-1] - portfolio_value_series.index[0]).total_seconds() / 3600.0
        if n_hours < 1:
            return 0.0
        annualized_return = (1 + total_return) ** (8760.0 / n_hours) - 1

        # Cálculo del Máximo Drawdown
        cumulative_max = portfolio_value_series.cummax()
        drawdown = (portfolio_value_series - cumulative_max) / cumulative_max
        max_drawdown = abs(drawdown.min())

        # Si el drawdown es menor al 1%, se considera insignificante y se penaliza
        # para evitar optimizaciones basadas en un riesgo irrealmente bajo.
        if max_drawdown < 0.01:
            return 0.0

        return annualized_return / max_drawdown

    def objective(self, trial):
        """
        Función objetivo que Optuna intentará maximizar.

        Implementa una validación cruzada Walk-Forward para evaluar la robustez
        de un conjunto de parámetros sugeridos por Optuna.

        Args:
            trial (optuna.trial.Trial): Un objeto de prueba de Optuna que sugiere
                                        los hiperparámetros a evaluar.

        Returns:
            float: El promedio del Calmar Ratio obtenido a través de los diferentes
                   segmentos (chunks) del Walk-Forward.
        """
        # 1. Sugerencia de hiperparámetros por parte de Optuna.
        params = {
            'ema_len': trial.suggest_int('ema_len', 50, 200),
            'macd_fast': trial.suggest_int('macd_fast', 7, 21),
            'macd_slow': trial.suggest_int('macd_slow', 22, 50),
            'macd_signal': trial.suggest_int('macd_signal', 7, 14),
            'adx_len': trial.suggest_int('adx_len', 10, 21),
            'adx_threshold': trial.suggest_int('adx_threshold', 22, 35),
            'stop_loss': trial.suggest_float('stop_loss', 0.02, 0.08, step=0.01),
            'take_profit': trial.suggest_float('take_profit', 0.04, 0.15, step=0.01),
            'n_shares': trial.suggest_float('n_shares', 0.1, 0.5, step=0.05)
        }

        indicator_params = {
            'ema_len': params['ema_len'], 'macd_fast': params['macd_fast'],
            'macd_slow': params['macd_slow'], 'macd_signal': params['macd_signal'],
            'adx_len': params['adx_len']
        }

        # Restricción lógica para los periodos del MACD.
        if params['macd_fast'] >= params['macd_slow']:
            return -1.0  # Penalización alta si la condición no se cumple.

        # 2. Lógica de Validación Walk-Forward.
        n_splits = 10
        len_data = len(self.data)
        chunk_size = len_data // n_splits
        objective_metrics = []

        if chunk_size < 30: # Asegura que cada segmento sea suficientemente grande.
            return -1.0

        for i in range(n_splits):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size
            chunk = self.data.iloc[start_idx:end_idx]

            chunk_with_indicators = add_indicators(chunk.copy(), **indicator_params)
            backtester = Backtester(chunk_with_indicators, params)
            portfolio_values = backtester.run(return_value_series=True)

            if not portfolio_values.empty:
                metric = self._calculate_objective_metric(portfolio_values)
                objective_metrics.append(metric)
            else:
                objective_metrics.append(-1.0) # Penalización si no se realizaron operaciones.

        if not objective_metrics:
            return -1.0

        # El valor a maximizar es el promedio de la métrica en todos los splits.
        return sum(objective_metrics) / len(objective_metrics)

    def run_optimization(self, n_trials=50):
        """
        Ejecuta el proceso de optimización con Optuna.

        Args:
            n_trials (int): El número de iteraciones que Optuna realizará para
                            buscar los mejores parámetros.

        Returns:
            dict: Un diccionario que contiene los mejores parámetros encontrados.
                  Retorna un diccionario vacío si no se encuentra una solución rentable.
        """
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)

        if study.best_value <= 0:
            print("Advertencia: No se encontró una combinación de parámetros rentable.")
            return {}

        print("\nProceso de optimización completado.")
        print(f"Mejor valor objetivo (Calmar Ratio Promedio): {study.best_value:.4f}")
        print("Parámetros óptimos encontrados:")
        for key, value in study.best_params.items():
            print(f"  - {key}: {value}")

        return study.best_params