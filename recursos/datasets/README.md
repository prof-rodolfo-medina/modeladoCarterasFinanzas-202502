# Datasets de Ejemplo

Esta carpeta contiene datos sintéticos para practicar la optimización de carteras.

## 📁 Archivos Disponibles

### 1. `activos_ejemplo.csv`
Información básica de 10 activos:
- Ticker (símbolo)
- Nombre de la empresa
- Sector
- Rendimiento esperado anualizado
- Volatilidad anualizada

### 2. `precios_historicos_completo.csv`
Precios diarios de 10 activos durante 252 días de trading (1 año).

### 3. `rendimientos_historicos.csv`
Rendimientos logarítmicos diarios calculados desde los precios.

### 4. `matriz_correlacion.csv`
Matriz de correlación entre los 10 activos.

### 5. `ejemplo_simple_3_activos.csv`
Dataset simplificado con 3 activos y 20 días.

### 6. `estadisticas_descriptivas.csv`
Estadísticas resumidas de cada activo.

---

## 🚀 Cómo Usar

### Ejemplo 1: Cargar precios
```python
import pandas as pd

precios = pd.read_csv('precios_historicos_completo.csv', 
                      index_col='fecha', parse_dates=True)
print(precios.head())
```

### Ejemplo 2: Calcular rendimientos
```python
import numpy as np

rendimientos = np.log(precios / precios.shift(1)).dropna()
rendimiento_anual = rendimientos.mean() * 252
volatilidad_anual = rendimientos.std() * np.sqrt(252)
```

### Ejemplo 3: Usar con EfficientFrontier
```python
from efficient_frontier import EfficientFrontier

rendimientos = pd.read_csv('rendimientos_historicos.csv', 
                           index_col='fecha', parse_dates=True)

expected_returns = rendimientos.mean() * 252
cov_matrix = rendimientos.cov() * 252

ef = EfficientFrontier(expected_returns, cov_matrix, risk_free_rate=0.03)
tangent = ef.tangent_portfolio()
print(f"Sharpe Ratio: {tangent['sharpe']:.4f}")
```

---

## 🔄 Generar Nuevos Datos

Para generar nuevos datasets con diferentes parámetros:
```bash
python generar_datos.py
```

---

## ⚠️ Nota Importante

Estos datos son **SINTÉTICOS** y se generaron para fines educativos. 
NO deben usarse para tomar decisiones de inversión reales.

Para datos reales, usa `yfinance` o fuentes financieras confiables.