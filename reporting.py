import quantstats as qs
import matplotlib.pyplot as plt
import pandas as pd

def generate_performance_report(portfolio_value_series, filename="performance_report.html"):
    """
    Genera un reporte de rendimiento completo en formato HTML utilizando la librería quantstats.

    Args:
        portfolio_value_series (pd.Series): Una serie de tiempo de Pandas que representa
                                            el valor del portafolio a lo largo del tiempo.
        filename (str): El nombre del archivo de salida para el reporte HTML.
    """
    # Quantstats opera sobre una serie de retornos, no sobre el valor absoluto del portafolio.
    # Por ello, primero se calcula el cambio porcentual.
    returns = portfolio_value_series.pct_change().dropna()

    print(f"  - Generando reporte de rendimiento HTML: {filename}...")
    try:
        qs.reports.html(returns, output=filename, title='Reporte de Desempeño de la Estrategia')
        print(f"  - Reporte '{filename}' guardado exitosamente.")
    except Exception as e:
        print(f"  - Error al generar el reporte de quantstats: {e}")


def plot_portfolio_value(portfolio_value_series, filename="equity_curve.png"):
    """
    Genera y guarda un gráfico de la curva de capital (valor del portafolio a lo largo del tiempo).

    Args:
        portfolio_value_series (pd.Series): La serie de tiempo del valor del portafolio.
        filename (str): El nombre del archivo de imagen de salida (ej. .png, .jpg).
    """
    print(f"  - Generando gráfico de curva de capital: {filename}...")
    try:
        plt.figure(figsize=(12, 6))
        portfolio_value_series.plot(title='Evolución del Valor del Portafolio (Curva de Capital)')
        plt.xlabel('Fecha')
        plt.ylabel('Valor del Portafolio ($)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        # Se cierra la figura para liberar memoria y evitar que se muestre en notebooks.
        plt.close()
        print(f"  - Gráfico '{filename}' guardado exitosamente.")
    except Exception as e:
        print(f"  - Error al generar el gráfico de la curva de capital: {e}")