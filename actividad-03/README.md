# Actividad 03: Optimización de Carteras Financieras

**Universidad Internacional de La Rioja (UNIR)**  
**Maestría en Ciencias Computacionales y Matemáticas Aplicadas**

---

## 📚 Contenido

Este módulo implementa la teoría moderna de carteras de Markowitz para la construcción de carteras óptimas de mínimo riesgo.

### Archivos Principales

- **`efficient_frontier.py`** - Clase principal para optimización de carteras
- **`requirements.txt`** - Dependencias necesarias
- **`README.md`** - Este archivo

---

## 🚀 Instalación
```bash
# Instalar dependencias
pip install -r requirements.txt
```

---

## 💡 Uso Básico
```python
from efficient_frontier import EfficientFrontier
import numpy as np

# Datos de ejemplo
rendimientos = np.array([0.12, 0.10, 0.14])
cov_matrix = np.array([
    [0.04, 0.01, 0.02],
    [0.01, 0.02, 0.01],
    [0.02, 0.01, 0.05]
])

# Crear optimizador
ef = EfficientFrontier(rendimientos, cov_matrix, risk_free_rate=0.03)

# Encontrar cartera tangente
tangent = ef.tangent_portfolio()
print(f"Sharpe Ratio: {tangent['sharpe']:.4f}")

# Visualizar frontera eficiente
ef.plot_efficient_frontier()
```

---

## 📖 Documentación

### Clase `EfficientFrontier`

#### Métodos Principales:

- **`min_variance_portfolio()`** - Cartera de mínima varianza
- **`tangent_portfolio()`** - Cartera tangente (máximo Sharpe)
- **`efficient_return(target_return)`** - Cartera óptima para rendimiento objetivo
- **`efficient_frontier_curve(n_points)`** - Puntos de la frontera eficiente
- **`plot_efficient_frontier()`** - Visualización completa

---

## 🎓 Para Estudiantes

Este código es la base para completar la **Actividad 02: Construcción de una Cartera Financiera de Mínimo Riesgo**.

### Pasos Sugeridos:

1. Estudia el código de `efficient_frontier.py`
2. Prueba con los datos de ejemplo
3. Modifica para usar tus propios activos
4. Documenta tus resultados

---

## 📞 Soporte

Para dudas sobre el código o la actividad, usa el **Foro "Pregúntale a tu profesor"** en el Aula Virtual Moodle.

---

**Última actualización:** Agosto 2025