"""
Script para generar datasets de ejemplo para la Actividad 02/03
Universidad Internacional de La Rioja (UNIR)

Este script genera datos sintéticos para pruebas y ejemplos
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generar_precios_correlacionados(n_days=252, n_assets=10, seed=42):
    """
    Genera precios sintéticos correlacionados.
    
    Parameters:
    -----------
    n_days : int
        Número de días de trading (252 = 1 año)
    n_assets : int
        Número de activos
    seed : int
        Semilla para reproducibilidad
    
    Returns:
    --------
    pd.DataFrame : DataFrame con precios
    """
    np.random.seed(seed)
    
    # Activos de ejemplo
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 
               'JPM', 'JNJ', 'XOM', 'WMT', 'PG'][:n_assets]
    
    # Parámetros de los activos
    mu = np.array([0.15, 0.12, 0.13, 0.18, 0.20, 
                   0.11, 0.09, 0.10, 0.08, 0.07])[:n_assets] / 252  # Diario
    
    sigma = np.array([0.25, 0.22, 0.20, 0.30, 0.35,
                      0.24, 0.16, 0.28, 0.18, 0.15])[:n_assets] / np.sqrt(252)  # Diario
    
    # Matriz de correlación (simplificada)
    correlation = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            if i < 5 and j < 5:  # Tech stocks más correlacionadas
                corr = np.random.uniform(0.6, 0.8)
            elif i >= 5 and j >= 5:  # Defensive stocks moderadamente correlacionadas
                corr = np.random.uniform(0.4, 0.6)
            else:  # Entre sectores menos correlación
                corr = np.random.uniform(0.2, 0.5)
            correlation[i, j] = corr
            correlation[j, i] = corr
    
    # Matriz de covarianzas
    cov_matrix = np.outer(sigma, sigma) * correlation
    
    # Generar rendimientos correlacionados
    returns = np.random.multivariate_normal(mu, cov_matrix, n_days)
    
    # Convertir a precios (partiendo de 100)
    prices = np.zeros((n_days + 1, n_assets))
    prices[0] = 100
    
    for t in range(1, n_days + 1):
        prices[t] = prices[t-1] * (1 + returns[t-1])
    
    # Crear fechas
    start_date = datetime(2024, 1, 2)
    dates = []
    current_date = start_date
    
    for _ in range(n_days + 1):
        # Saltar fines de semana
        while current_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
            current_date += timedelta(days=1)
        dates.append(current_date)
        current_date += timedelta(days=1)
    
    # Crear DataFrame
    df_prices = pd.DataFrame(prices, columns=tickers, index=dates)
    df_prices.index.name = 'fecha'
    
    return df_prices


def generar_rendimientos_desde_precios(df_prices):
    """Calcula rendimientos logarítmicos desde precios."""
    returns = np.log(df_prices / df_prices.shift(1))
    return returns.dropna()


def guardar_datasets():
    """Genera y guarda todos los datasets de ejemplo."""
    
    print("🔄 Generando datasets de ejemplo...")
    
    # 1. Precios históricos completos (10 activos, 1 año)
    print("   📊 Generando precios históricos (252 días)...")
    df_prices_full = generar_precios_correlacionados(n_days=252, n_assets=10, seed=42)
    df_prices_full.to_csv('precios_historicos_completo.csv')
    print(f"      ✓ Guardado: precios_historicos_completo.csv ({len(df_prices_full)} filas)")
    
    # 2. Rendimientos desde precios
    print("   📈 Calculando rendimientos...")
    df_returns = generar_rendimientos_desde_precios(df_prices_full)
    df_returns.to_csv('rendimientos_historicos.csv')
    print(f"      ✓ Guardado: rendimientos_historicos.csv ({len(df_returns)} filas)")
    
    # 3. Dataset simple (3 activos, 20 días)
    print("   📉 Generando dataset simple (3 activos)...")
    df_simple = generar_precios_correlacionados(n_days=20, n_assets=3, seed=123)
    df_simple.columns = ['Accion_A', 'Accion_B', 'Accion_C']
    df_simple.to_csv('ejemplo_simple_3_activos.csv')
    print(f"      ✓ Guardado: ejemplo_simple_3_activos.csv ({len(df_simple)} filas)")
    
    # 4. Información de activos
    print("   📋 Generando información de activos...")
    activos_info = pd.DataFrame({
        'ticker': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 
                   'JPM', 'JNJ', 'XOM', 'WMT', 'PG'],
        'nombre': ['Apple Inc.', 'Alphabet Inc.', 'Microsoft Corp.', 
                   'Amazon.com Inc.', 'Tesla Inc.', 'JPMorgan Chase',
                   'Johnson & Johnson', 'Exxon Mobil', 'Walmart Inc.', 
                   'Procter & Gamble'],
        'sector': ['Tecnología', 'Tecnología', 'Tecnología', 
                   'Consumo Discrecional', 'Consumo Discrecional', 
                   'Financiero', 'Salud', 'Energía', 
                   'Consumo Básico', 'Consumo Básico'],
        'rendimiento_esperado': [0.15, 0.12, 0.13, 0.18, 0.20, 
                                 0.11, 0.09, 0.10, 0.08, 0.07],
        'volatilidad': [0.25, 0.22, 0.20, 0.30, 0.35, 
                       0.24, 0.16, 0.28, 0.18, 0.15]
    })
    activos_info.to_csv('activos_ejemplo.csv', index=False)
    print(f"      ✓ Guardado: activos_ejemplo.csv ({len(activos_info)} filas)")
    
    # 5. Matriz de correlación
    print("   🔗 Generando matriz de correlación...")
    returns_array = df_returns.values
    corr_matrix = pd.DataFrame(
        np.corrcoef(returns_array.T),
        index=df_returns.columns,
        columns=df_returns.columns
    )
    corr_matrix.to_csv('matriz_correlacion.csv')
    print(f"      ✓ Guardado: matriz_correlacion.csv ({corr_matrix.shape[0]}x{corr_matrix.shape[1]})")
    
    # 6. Estadísticas descriptivas
    print("   📊 Generando estadísticas descriptivas...")
    stats = df_returns.describe().T
    stats['rendimiento_anual'] = df_returns.mean() * 252
    stats['volatilidad_anual'] = df_returns.std() * np.sqrt(252)
    stats.to_csv('estadisticas_descriptivas.csv')
    print(f"      ✓ Guardado: estadisticas_descriptivas.csv")
    
    print("\n✅ Todos los datasets generados exitosamente!")
    print("\n📁 Archivos creados:")
    print("   - precios_historicos_completo.csv")
    print("   - rendimientos_historicos.csv")
    print("   - ejemplo_simple_3_activos.csv")
    print("   - activos_ejemplo.csv")
    print("   - matriz_correlacion.csv")
    print("   - estadisticas_descriptivas.csv")


if __name__ == "__main__":
    print("=" * 70)
    print("GENERADOR DE DATASETS DE EJEMPLO")
    print("Actividad 02/03: Construcción de Carteras de Mínimo Riesgo")
    print("=" * 70)
    print()
    
    guardar_datasets()
    
    print("\n" + "=" * 70)
    print("USO:")
    print("Los estudiantes pueden usar estos archivos directamente o")
    print("ejecutar este script para generar nuevos datos con diferentes semillas.")
    print("=" * 70)