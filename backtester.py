import pandas as pd
from config import INITIAL_CASH, COMMISSION


class Backtester:
    """
    Motor de backtesting para simular una estrategia de trading sobre datos históricos.

    Esta clase encapsula la lógica para:
    1. Generar señales de compra y venta basadas en una estrategia de indicadores.
    2. Simular la ejecución de operaciones, incluyendo comisiones, stop-loss y take-profit.
    3. Calcular la evolución del valor del portafolio a lo largo del tiempo.

    Attributes:
        data (pd.DataFrame): DataFrame con los datos de mercado y los indicadores ya calculados.
        params (dict): Diccionario con los parámetros optimizados para la estrategia.
        initial_cash (float): Capital inicial para la simulación.
        commission (float): Costo por operación (comisión del broker).
        price_col (str): Nombre de la columna de precios a utilizar para la simulación (ej. 'close').
    """

    def __init__(self, data, params):
        """
        Inicializa el motor de backtesting.

        Args:
            data (pd.DataFrame): Los datos de mercado con indicadores.
            params (dict): Los parámetros de la estrategia.
        """
        self.data = data.copy()
        self.params = params
        self.initial_cash = INITIAL_CASH
        self.commission = COMMISSION
        self.price_col = 'close'

    def _generate_signals(self):
        """
        Metodo privado para generar las señales de entrada de la estrategia.

        Implementa una regla de "2 de 3 condiciones" para mayor robustez:
        - Condición 1: Tendencia (basada en EMA).
        - Condición 2: Fuerza de la tendencia (basada en ADX).
        - Condición 3: Momento (cruce de MACD).

        Las señales se añaden como nuevas columnas ('buy_signal', 'sell_signal')
        al DataFrame self.data.
        """
        # --- Extraer parámetros para mayor legibilidad ---
        adx_threshold = self.params.get('adx_threshold')
        ema_len = self.params.get('ema_len')
        adx_len = self.params.get('adx_len')
        macd_fast = self.params.get('macd_fast')
        macd_slow = self.params.get('macd_slow')
        macd_signal = self.params.get('macd_signal')

        # --- Nombres de columna dinámicos basados en parámetros ---
        ema_col = f"EMA_{ema_len}"
        adx_col = f"ADX_{adx_len}"
        macd_line_col = f"MACD_{macd_fast}_{macd_slow}_{macd_signal}"
        macd_signal_col = f"MACDs_{macd_fast}_{macd_slow}_{macd_signal}"

        # --- Definición de las condiciones de la estrategia ---
        cond_ema_up = self.data[self.price_col] > self.data[ema_col]
        cond_ema_down = self.data[self.price_col] < self.data[ema_col]
        cond_adx_strong = self.data[adx_col] > adx_threshold

        # Para evitar lookahead bias, se usa .shift(1) para comparar el cruce
        # con los valores de la vela anterior.
        cond_macd_cross_up = (self.data[macd_line_col].shift(1) < self.data[macd_signal_col].shift(1)) & \
                             (self.data[macd_line_col] > self.data[macd_signal_col])
        cond_macd_cross_down = (self.data[macd_line_col].shift(1) > self.data[macd_signal_col].shift(1)) & \
                               (self.data[macd_line_col] < self.data[macd_signal_col])

        # --- Lógica de "2 de 3": se genera una señal si al menos dos condiciones son verdaderas ---
        self.data['buy_signal'] = (cond_ema_up.astype(int) +
                                   cond_adx_strong.astype(int) +
                                   cond_macd_cross_up.astype(int)) >= 2

        self.data['sell_signal'] = (cond_ema_down.astype(int) +
                                    cond_adx_strong.astype(int) +
                                    cond_macd_cross_down.astype(int)) >= 2

    def run(self, return_value_series=False):
        """
        Ejecuta la simulación de backtesting.

        Itera a través de los datos vela por vela, aplicando la lógica de
        apertura y cierre de posiciones.

        Args:
            return_value_series (bool): Si es True, retorna una serie de Pandas
                                        con el valor del portafolio en cada
                                        punto del tiempo. Si es False, retorna
                                        solo el valor final del portafolio.
        Returns:
            pd.Series or float: La serie de tiempo del valor del portafolio o el valor final.
        """
        # --- Extraer parámetros de gestión de riesgo ---
        stop_loss_pct = self.params.get('stop_loss')
        take_profit_pct = self.params.get('take_profit')
        n_shares = self.params.get('n_shares')  # Fracción del capital a arriesgar

        self._generate_signals()

        # --- Inicialización de variables de la simulación ---
        cash = self.initial_cash
        position = 0.0  # Cantidad de activo en posesión (>0 para largo, <0 para corto)
        entry_price = 0.0
        portfolio_values = []

        # --- Bucle principal de la simulación ---
        for i in range(1, len(self.data)):
            current_price = self.data[self.price_col].iloc[i]

            # 1. Lógica de CIERRE de posición (por Stop-Loss o Take-Profit)
            if position > 0:  # Si estamos en una posición larga (comprado)
                stop_loss_price = entry_price * (1 - stop_loss_pct)
                take_profit_price = entry_price * (1 + take_profit_pct)
                if current_price <= stop_loss_price or current_price >= take_profit_price:
                    cash += position * current_price * (1 - self.commission)
                    position = 0.0

            elif position < 0:  # Si estamos en una posición corta (vendido)
                stop_loss_price = entry_price * (1 + stop_loss_pct)
                take_profit_price = entry_price * (1 - take_profit_pct)
                if current_price >= stop_loss_price or current_price <= take_profit_price:
                    cash -= abs(position) * current_price * (1 + self.commission)
                    position = 0.0

            # 2. Lógica de APERTURA de posición (si no hay una posición abierta)
            if position == 0:
                if self.data['buy_signal'].iloc[i]:
                    investment_amount = cash * n_shares
                    cost = investment_amount * (1 + self.commission)
                    if cash > cost:  # Asegurarse de tener suficiente capital
                        cash -= cost
                        position = investment_amount / current_price
                        entry_price = current_price

                elif self.data['sell_signal'].iloc[i]:
                    investment_amount = cash * n_shares
                    proceeds = investment_amount * (1 - self.commission)
                    cash += proceeds
                    position = - (investment_amount / current_price)
                    entry_price = current_price

            # 3. Cálculo del valor total del portafolio en el tiempo actual
            portfolio_value = cash
            if position > 0:
                portfolio_value += position * current_price
            elif position < 0:
                # CORRECCIÓN IMPORTANTE: El valor del portafolio en una posición corta
                # es el efectivo total MENOS el costo actual para cerrar la posición (la obligación).
                portfolio_value = cash - (abs(position) * current_price)

            portfolio_values.append(portfolio_value)

        # --- Retorno de resultados ---
        if return_value_series:
            return pd.Series(portfolio_values, index=self.data.index[1:])

        return portfolio_values[-1] if portfolio_values else self.initial_cash