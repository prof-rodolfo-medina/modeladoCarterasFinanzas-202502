# Ejercicio 1: Implementación Básica del Algoritmo Hull-White

**Nivel:** Básico  
**Tiempo estimado:** 2-3 horas  
**Objetivos:** Comprender e implementar el algoritmo Hull-White desde cero

## 🎯 Objetivos de Aprendizaje

Al completar este ejercicio, serás capaz de:
1. Implementar el algoritmo Hull-White paso a paso
2. Validar parámetros calibrados
3. Comparar resultados con implementación de referencia
4. Interpretar resultados económicamente

## 📋 Tareas a Realizar

### Parte A: Implementación Manual (40 puntos)

#### A.1 Cálculo de Rendimientos (10 puntos)
Dada la serie de precios: `[100, 105, 98, 103, 110, 95, 108, 102, 115]`

**Tareas:**
1. Calcular manualmente los rendimientos relativos $U_i = \frac{S_i}{S_{i-1}} - 1$
2. Presentar los cálculos en una tabla
3. Verificar tus cálculos con código

**Tabla a completar:**
| i | Si-1 | Si | Ui = (Si/Si-1) - 1 |
|---|------|----|--------------------|
| 1 | 100  | 105| ?                  |
| 2 | 105  | 98 | ?                  |
| ... | ... | ... | ...              |

#### A.2 Estadísticas Descriptivas (10 puntos)
Con los rendimientos calculados:

1. **Media muestral:** $\bar{U} = \frac{1}{n}\sum_{i=1}^n U_i$
2. **Desviación estándar muestral:** $S_U = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (U_i - \bar{U})^2}$

**Preguntas:**
- ¿Qué representa económicamente $\bar{U}$?
- ¿Qué representa $S_U$ en términos de riesgo?

#### A.3 Calibración Hull-White (10 puntos)
Aplicar las fórmulas:
- $u = 1 + \bar{U} + S_U$
- $d = 1 + \bar{U} - S_U$

**Validación:**
- ¿Se cumple que $0 < d < 1 < u$?
- ¿Qué significa si alguna condición no se cumple?

#### A.4 Interpretación Económica (10 puntos)
**Preguntas a responder:**
1. Si $u = 1.08$ y $d = 0.94$, ¿qué significa esto para el activo?
2. ¿Cómo se relaciona el ratio $u/d$ con la volatilidad?
3. ¿Por qué Hull-White asume $p = 0.5$?

### Parte B: Implementación en Código (40 puntos)

#### B.1 Función Básica (20 puntos)

Implementa una función que calcule u y d:

**Python:**
```python
def hull_white_basic(prices):
    """
    Implementación básica del algoritmo Hull-White
    
    Args:
        prices (list): Lista de precios históricos
        
    Returns:
        dict: Diccionario con u, d, mu, sigma
    """
    # Tu código aquí
    pass

# Ejemplo de uso
prices = [100, 105, 98, 103, 110, 95, 108, 102, 115]
result = hull_white_basic(prices)
print(f"u = {result['u']:.6f}")
print(f"d = {result['d']:.6f}")
```

**MATLAB:**
```matlab
function results = hullWhiteBasic(prices)
% HULLWHITEBASIC Implementación básica del algoritmo Hull-White
%
% Input:
%   prices - vector de precios históricos
%
% Output:
%   results - estructura con u, d, mu, sigma

    % Tu código aquí
    
end
```

#### B.2 Validación y Pruebas (20 puntos)

1. **Validar con datos del ejercicio:**
   - Usa los precios del Long Straddle: `[100, 102, 98, 105, 103, 99, 107, 104]`
   - Compara tus resultados con la implementación completa

2. **Casos extremos:**
   - ¿Qué pasa con precios constantes? `[100, 100, 100, 100]`
   - ¿Qué pasa con alta volatilidad? `[100, 150, 75, 120, 80]`

3. **Pruebas de robustez:**
   - Mínimo número de observaciones necesarias
   - Manejo de precios negativos o cero

### Parte C: Aplicación a Valoración (20 puntos)

#### C.1 Valoración de Opción Simple (15 puntos)

Usando tus parámetros calibrados, valora una **opción call europea** con:
- $S_0 = 100$, $K = 105$, $T = 2$ períodos, $r = 0.05$

**Pasos:**
1. Calcular probabilidad risk-neutral: $q^* = \frac{e^r - d}{u - d}$
2. Construir árbol de precios para 2 períodos
3. Calcular payoffs finales: $\max(S_T - K, 0)$
4. Trabajar hacia atrás para obtener precio de la opción

#### C.2 Análisis de Resultados (5 puntos)

**Preguntas:**
1. ¿Cómo se compara tu precio con Black-Scholes? (puedes usar una calculadora online)
2. ¿Qué efecto tiene cambiar $u$ y $d$ en el precio de la opción?
3. ¿Es realista la probabilidad risk-neutral calculada?

## 🔍 Entregables

### Formato de Entrega
1. **Archivo principal:** `apellido_nombre_ejercicio1.pdf`
2. **Código:** `apellido_nombre_codigo1.py` o `apellido_nombre_codigo1.m`
3. **Cálculos manuales:** Incluidos en el PDF o en archivo Excel separado

### Contenido del Reporte
1. **Portada** con datos del estudiante
2. **Cálculos manuales** de la Parte A con tablas claras
3. **Código comentado** de la Parte B
4. **Resultados numéricos** de la Parte C
5. **Análisis e interpretación** de resultados
6. **Conclusiones** sobre el algoritmo Hull-White

## ✅ Criterios de Evaluación

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| **Cálculos manuales** | 40 | Precisión en cálculos, tablas claras, interpretación correcta |
| **Implementación** | 40 | Código funcional, manejo de errores, pruebas válidas |
| **Aplicación** | 20 | Valoración correcta, análisis de resultados |
| **Presentación** | Extra | Claridad, orden, documentación del código |

## 🚨 Errores Comunes a Evitar

1. **Error en rendimientos:** Usar $\frac{S_i - S_{i-1}}{S_i}$ en lugar de $\frac{S_i - S_{i-1}}{S_{i-1}}$
2. **Error en desviación estándar:** Usar $n$ en lugar de $n-1$ en el denominador
3. **No validar parámetros:** No verificar que $u > d > 0$
4. **Error en probabilidad risk-neutral:** Usar $p = 0.5$ en lugar de $q^*$

## 💡 Pistas y Sugerencias

### Para los Cálculos Manuales
- Usa una calculadora con suficientes decimales
- Mantén al menos 6 decimales en cálculos intermedios
- Verifica tus cálculos sumando: $\sum U_i$ debe ser igual a $n \times \bar{U}$

### Para la Implementación
- Usa `numpy` en Python para cálculos vectorizados
- En MATLAB, aprovecha funciones built-in como `mean()` y `std()`
- Implementa validación de entrada antes de los cálculos

### Para la Interpretación
- Piensa en $u$ y $d$ como factores multiplicativos
- El ratio $u/d$ indica la "amplitud" de movimientos posibles
- Una alta volatilidad histórica resulta en mayor diferencia entre $u$ y $d$

## 📚 Material de Consulta

### Fórmulas Clave
```
Rendimiento relativo: Ui = (Si/Si-1) - 1
Media muestral: μ̄ = (1/n) × Σ Ui
Desviación estándar: σ = √[(1/(n-1)) × Σ(Ui - μ̄)²]
Factor de subida: u = 1 + μ̄ + σ
Factor de bajada: d = 1 + μ̄ - σ
Probabilidad risk-neutral: q* = (e^r - d)/(u - d)
```

### Datos de Prueba
```python
# Serie básica del ejercicio
prices_basic = [100, 105, 98, 103, 110, 95, 108, 102, 115]

# Serie del Long Straddle
prices_straddle = [100, 102, 98, 105, 103, 99, 107, 104]

# Serie con alta volatilidad
prices_volatile = [100, 150, 75, 120, 80, 160, 90]

# Serie estable
prices_stable = [100, 101, 99, 102, 98, 103, 97]
```

## 🎓 Extensiones Opcionales (Puntos Extra)

### Extensión 1: Análisis Gráfico (5 puntos)
Crear gráficos que muestren:
1. Serie de precios original
2. Serie de rendimientos
3. Histograma de rendimientos con distribución normal superpuesta

### Extensión 2: Comparación de Métodos (5 puntos)
Comparar Hull-White con:
1. Parámetros fijos dados ($u = 1.1$, $d = 0.9$)
2. Método de máxima verosimilitud
3. Calibración por volatilidad implícita (investigación)

### Extensión 3: Simulación Monte Carlo (10 puntos)
1. Simular 1000 trayectorias de precios usando tus parámetros $u$ y $d$
2. Comparar la distribución simulada con los datos originales
3. Validar que la volatilidad simulada coincide con $\sigma$ calculado

## 🔗 Recursos Adicionales

### Lecturas Recomendadas
- Hull, J. C. (2021). *Options, Futures, and Other Derivatives*, Capítulo 13
- Apuntes del curso: Tema 4, Secciones 4.2-4.4

### Herramientas Online
- **Calculadora Black-Scholes:** [calculator.net/black-scholes-calculator](https://calculator.net/black-scholes-calculator.html)
- **Verificador de cálculos:** Usa Excel o Google Sheets para validar

### Códigos de Referencia
- Implementación completa disponible en el repositorio
- Usa solo para verificar resultados, no para copiar

## 📝 Template de Código

### Python Starter
```python
import numpy as np
import matplotlib.pyplot as plt

def hull_white_basic(prices):
    """Tu implementación aquí"""
    
    # 1. Validar entrada
    if len(prices) < 2:
        raise ValueError("Se necesitan al menos 2 precios")
    
    # 2. Calcular rendimientos
    # Tu código aquí
    
    # 3. Estadísticas descriptivas
    # Tu código aquí
    
    # 4. Calibración Hull-White
    # Tu código aquí
    
    # 5. Validación
    # Tu código aquí
    
    return {
        'u': u,
        'd': d,
        'mu': mu,
        'sigma': sigma,
        'is_valid': is_valid
    }

# Pruebas
if __name__ == "__main__":
    prices = [100, 105, 98, 103, 110, 95, 108, 102, 115]
    result = hull_white_basic(prices)
    print("Resultados Hull-White:")
    for key, value in result.items():
        print(f"{key}: {value}")
```

### MATLAB Starter
```matlab
function results = hullWhiteBasic(prices)
    % Validar entrada
    if length(prices) < 2
        error('Se necesitan al menos 2 precios');
    end
    
    % Calcular rendimientos
    % Tu código aquí
    
    % Estadísticas descriptivas  
    % Tu código aquí
    
    % Calibración Hull-White
    % Tu código aquí
    
    % Validación
    % Tu código aquí
    
    % Estructura de resultados
    results.u = u;
    results.d = d;
    results.mu = mu;
    results.sigma = sigma;
    results.is_valid = is_valid;
end
```

## ⏰ Cronograma Sugerido

### Día 1 (1-2 horas)
- Cálculos manuales de la Parte A
- Revisión de conceptos teóricos

### Día 2 (1-2 horas)  
- Implementación básica del código
- Pruebas con datos simples

### Día 3 (1 hora)
- Aplicación a valoración de opciones
- Análisis de resultados y conclusiones

## 🤝 Política de Colaboración

- ✅ **Permitido:** Discutir conceptos teóricos con compañeros
- ✅ **Permitido:** Usar material del curso y referencias académicas
- ✅ **Permitido:** Consultar documentación de Python/MATLAB
- ❌ **No permitido:** Copiar código de compañeros
- ❌ **No permitido:** Usar implementaciones completas sin entender

## 📞 Ayuda y Consultas

### Dudas Frecuentes
**P: ¿Qué hacer si obtengo d < 0?**
R: Revisa tus cálculos. Si son correctos, significa que la volatilidad es muy alta para el modelo básico.

**P: ¿Es normal que u y d sean muy cercanos a 1?**
R: Sí, si la volatilidad histórica es baja. El modelo refleja movimientos pequeños.

**P: ¿Cómo interpreto una probabilidad risk-neutral mayor a 1?**
R: Indica que los parámetros no son consistentes con arbitraje. Revisa la calibración.

### Contacto
- **Foro del curso:** Para dudas conceptuales
- **Horas de oficina:** Para revisión de código
- **Email:** Para consultas urgentes

---

**¡Éxito en tu implementación! Recuerda que el objetivo es entender profundamente el algoritmo, no solo obtener números correctos.** 🚀