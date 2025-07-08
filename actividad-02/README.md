# Actividad 02: Algoritmo Hull-White para Calibración de Parámetros Binomiales

> **Curso:** Modelización y Valoración de Derivados y Carteras en Finanzas  
> **Fecha de Publicación:** 08 Jul 2025  
> **Fecha de Entrega:** 15 Jul 2025  
> **Estado:** ✅ Disponible

## 🎯 Objetivos de Aprendizaje

Al completar esta actividad, serás capaz de:

- ✅ **Comprender** la fundamentación teórica del algoritmo Hull-White
- ✅ **Implementar** el método para calibrar parámetros u y d desde datos históricos
- ✅ **Aplicar** el algoritmo en Python y MATLAB para valoración de opciones
- ✅ **Analizar** la sensibilidad y robustez de los parámetros calibrados
- ✅ **Valorar** estrategias de opciones usando parámetros empíricos

## 📚 Contenido Teórico

### ¿Qué es el Algoritmo Hull-White?

El algoritmo Hull-White permite calibrar los parámetros **u** (factor de subida) y **d** (factor de bajada) de un modelo binomial utilizando datos históricos del activo subyacente, proporcionando una base empírica sólida para la valoración de opciones.

### Fundamentos Matemáticos

El algoritmo se basa en igualar los momentos del modelo binomial con las estadísticas observadas:

```
E[S(t+1)/S(t)] = 1 + μΔt
V[S(t+1)/S(t)] = σ²Δt
```

Con la simplificación de Hull-White (p = 0.5):

```
u = 1 + μΔt + σ√Δt
d = 1 + μΔt - σ√Δt
```

## 📁 Estructura de la Actividad

```
📦 actividad-02/
├── 📄 README.md                    # Este archivo
├── 📄 QUICK_START.md               # Inicio rápido
├── 📂 codigo/                      # Implementaciones completas
│   ├── 🐍 hull_white_python.py     # Código Python con clases
│   ├── 🔷 hull_white_matlab.m      # Código MATLAB modular
│   ├── 📊 data_loader.py           # Cargador de datos
│   └── 📋 requirements.txt         # Dependencias Python
├── 📂 presentacion/                # Material de clase
│   ├── 📊 hull_white_slides.pdf    # Presentación Beamer
│   └── 📖 bibliografia.bib         # Referencias académicas
├── 📂 ejercicios/                  # Tareas graduales
│   ├── 📝 ejercicio_basico.md      # Implementación básica
│   ├── 📈 ejercicio_intermedio.md  # Aplicación avanzada
│   └── 🔬 ejercicio_avanzado.md    # Análisis comparativo
└── 📂 datos/                       # Datasets de ejemplo
    ├── 📊 precios_ejemplo.csv      # Serie sintética
    └── 💹 datos_long_straddle.csv  # Datos del ejercicio anterior
```

## 🚀 Inicio Rápido

### Prerrequisitos
- Python 3.8+ con librerías: `numpy`, `pandas`, `matplotlib`, `scipy`
- MATLAB R2020a+ (con Statistics Toolbox)
- Conocimientos de la Actividad 01 (Long Straddle)

### Instalación

```bash
# 1. Navegar a la actividad
cd actividad-02/

# 2. Instalar dependencias Python
pip install -r codigo/requirements.txt

# 3. Verificar instalación
python codigo/hull_white_python.py
```

### MATLAB Setup
```matlab
% Agregar carpeta al path
addpath('actividad-02/codigo/');

% Verificar instalación
which hullWhiteCalibration
```

## 📊 Manejo de Datos

### Data Loader Integrado

La actividad incluye un **cargador de datos inteligente** que facilita el trabajo con diferentes datasets:

```python
from codigo.data_loader import DataLoader

# Inicializar cargador
loader = DataLoader()

# Cargar datos principales
precios = loader.load_sample_prices()          # Serie de 50 precios de ejemplo
straddle = loader.load_straddle_data()         # Datos del Long Straddle
test_data = loader.get_hull_white_test_data()  # Datasets de prueba predefinidos
```

### Datasets Disponibles

| Dataset | Archivo | Descripción | Uso Recomendado |
|---------|---------|-------------|-----------------|
| **Precios Ejemplo** | `precios_ejemplo.csv` | 50 observaciones con volatilidad realista | Calibración principal y análisis |
| **Long Straddle** | `datos_long_straddle.csv` | 24 días de opciones call/put | Conexión con Actividad 01 |
| **Series de Prueba** | *Generadas en código* | 7 series predefinidas | Testing y validación |

### Casos de Uso Específicos

#### 1. Calibración Básica
```python
# Cargar datos y extraer solo precios
df_precios = loader.load_sample_prices()
precios_lista = df_precios['precio'].tolist()

# Usar con Hull-White
from hull_white_python import HullWhiteCalibrator
calibrator = HullWhiteCalibrator(precios_lista)
```

#### 2. Análisis Comparativo
```python
# Obtener múltiples series para comparar
test_series = loader.get_hull_white_test_data()

for nombre, precios in test_series.items():
    calibrator = HullWhiteCalibrator(precios)
    params = calibrator.calibrate_hull_white()
    print(f"{nombre}: u={params['u']:.4f}, d={params['d']:.4f}")
```

#### 3. Conexión con Long Straddle
```python
# Cargar datos de opciones
straddle_df = loader.load_straddle_data()
precios_subyacente = straddle_df['precio_subyacente'].tolist()

# Calibrar y comparar con precios de mercado
hw_params = HullWhiteCalibrator(precios_subyacente).calibrate_hull_white()
# ... resto del análisis
```

### Generación Automática

Si los archivos CSV no existen, el data loader los **genera automáticamente**:

```python
# Crear archivos faltantes
loader.create_sample_files()
# ✅ Crea precios_ejemplo.csv y datos_long_straddle.csv
```

## 💻 Ejemplos de Uso

### Python - Calibración Básica

```python
from codigo.hull_white_python import HullWhiteCalibrator
from codigo.data_loader import DataLoader

# Cargar datos con el data loader
loader = DataLoader()
df_precios = loader.load_sample_prices()
precios = df_precios['precio'].tolist()

# Calibrar parámetros
calibrador = HullWhiteCalibrator(precios)
parametros = calibrador.calibrate_hull_white()

print(f"Factor u: {parametros['u']:.6f}")
print(f"Factor d: {parametros['d']:.6f}")
print(f"Modelo válido: {parametros['is_valid']}")

# Análisis gráfico completo
calibrador.plot_analysis()
```

### MATLAB - Análisis Completo

```matlab
% Cargar datos desde CSV
datos = readtable('datos/precios_ejemplo.csv');
precios = datos.precio;

% Calibración completa con análisis
resultados = hullWhiteCalibration(precios, 'Verbose', true);

% Generar visualizaciones
plotHullWhiteAnalysis(resultados);

% Análisis de sensibilidad
sensibilidad = sensitivityAnalysis(precios);
plotSensitivityAnalysis(sensibilidad);
```

## 🔗 Conexión con Actividad 01

Esta actividad extiende el **análisis Long Straddle** de la Actividad 01:

```python
# En lugar de usar parámetros dados (C=5€, P=4€, E=100€)
# Ahora calibramos u y d desde datos históricos

from codigo.hull_white_python import OptionPricer
from codigo.data_loader import DataLoader

# Cargar datos del straddle
loader = DataLoader()
straddle_df = loader.load_straddle_data()
precios_subyacente = straddle_df['precio_subyacente'].tolist()

# Calibrar parámetros
calibrador = HullWhiteCalibrator(precios_subyacente)
hw_params = calibrador.calibrate_hull_white()

# Valorar Long Straddle con parámetros empíricos
pricer = OptionPricer(hw_params)
straddle = pricer.long_straddle_analysis(
    S0=100, K=100, T=3, r=0.05,
    market_call=5.0, market_put=4.0
)

print(f"Call teórica: {straddle['call_value']:.4f} vs Mercado: 5.00")
print(f"Put teórica: {straddle['put_value']:.4f} vs Mercado: 4.00")
```

## 📝 Ejercicios y Evaluación

| Ejercicio | Descripción | Puntos | Nivel |
|-----------|-------------|--------|-------|
| **Básico** | Implementación manual del algoritmo | 40 | 🟢 Principiante |
| **Intermedio** | Aplicación a datos reales y comparación | 35 | 🟡 Intermedio |
| **Avanzado** | Análisis comparativo y extensiones | 25 | 🔴 Avanzado |

### Criterios de Evaluación
- **Precisión técnica** (40%): Implementación correcta del algoritmo
- **Análisis crítico** (30%): Interpretación económica de resultados
- **Presentación** (20%): Claridad en código y documentación
- **Innovación** (10%): Extensiones o mejoras propuestas

## 🔬 Casos de Estudio

### Caso 1: Recalibración Long Straddle
Usar datos históricos para recalibrar los parámetros del Long Straddle y comparar con precios de mercado.

### Caso 2: Análisis de Robustez
Estudiar cómo afecta el tamaño de la ventana temporal en la estabilidad de parámetros.

### Caso 3: Comparación de Métodos
Contrastar Hull-White estándar vs. pesos exponenciales vs. probabilidades no simétricas.

## 📈 Resultados Esperados

Al finalizar esta actividad, tendrás:

1. **Implementación funcional** del algoritmo en ambos lenguajes
2. **Comprensión profunda** de la calibración de parámetros
3. **Herramientas de análisis** para validar modelos binomiales
4. **Conexión práctica** entre teoría y aplicación empírica

## 🔧 Troubleshooting

### Problemas Comunes

**❌ Error: "d ≤ 0" o "u ≤ d"**
```python
# Solución: Verificar datos de entrada y considerar filtros
if parametros['d'] <= 0:
    print("⚠️ Volatilidad muy alta para el modelo básico")
    print("💡 Considerar filtrar outliers o usar ventana más amplia")
```

**❌ Error: "Muy pocos datos"**
```python
# Solución: Asegurar mínimo de observaciones
if len(precios) < 10:
    print("⚠️ Se recomiendan al menos 20 observaciones para estabilidad")
```

**❌ Error: "Módulo data_loader no encontrado"**
```python
# Solución: Verificar ubicación y path
import sys
from pathlib import Path
sys.path.append(str(Path.cwd() / "codigo"))
from data_loader import DataLoader
```

### Validación de Resultados
```python
# Verificar coherencia económica
def validar_parametros(u, d):
    checks = {
        'd_positivo': d > 0,
        'u_mayor_d': u > d,
        'u_crecimiento': u > 1,
        'ratio_razonable': 1.1 < u/d < 2.0
    }
    return all(checks.values()), checks

# Usar con tus resultados
es_valido, detalles = validar_parametros(parametros['u'], parametros['d'])
```

## 📚 Material de Apoyo

### Lecturas Recomendadas
- **Hull, J. C. (2021).** *Options, Futures, and Other Derivatives*, Cap. 13
- **Tema 4 del Curso:** Valoración con árboles multi-periodo
- **Paper original:** Hull & White (1988) - Control Variate Technique

### Videos y Tutoriales
- 🎥 **Sesión grabada:** Derivación matemática del algoritmo
- 🎥 **Demo en vivo:** Implementación paso a paso
- 🎥 **Casos prácticos:** Aplicación a diferentes mercados

### Herramientas Adicionales
- 🧮 **Calculadora online:** Verificación de Black-Scholes
- 📊 **Datasets adicionales:** Más series históricas para pruebas
- 🔗 **APIs financieras:** Código para descargar datos reales

## 📞 Soporte y Consultas

### Métodos de Contacto
- 💬 **Foro del curso:** Para dudas conceptuales y técnicas
- 📧 **Email directo:** `prof-rodolfo-medina` para consultas urgentes
- 🕐 **Horario de consultas:** 48h después de publicar en el foro

### FAQs

**P: ¿Puedo usar datos de Yahoo Finance?**  
R: ¡Sí! Hay código de ejemplo para descargar datos reales.

**P: ¿Qué hacer si MATLAB no tiene la Financial Toolbox?**  
R: El código está diseñado para funcionar solo con Statistics Toolbox.

**P: ¿Es normal que los parámetros cambien mucho con pocos datos?**  
R: Sí, es esperado. Por eso se recomienda análisis de sensibilidad.

**P: ¿Los archivos CSV se crean automáticamente?**  
R: Sí, ejecuta `loader.create_sample_files()` si no existen.

**P: ¿Cómo conecto esto con la Actividad 01?**  
R: Usa `loader.load_straddle_data()` para obtener datos compatibles.

---

## 🚀 ¿Listo para empezar?

1. **📖 Revisa** la presentación en `presentacion/hull_white_slides.pdf`
2. **⚡ Inicio rápido:** Consulta `QUICK_START.md` para comenzar inmediatamente
3. **💻 Ejecuta** los ejemplos básicos para familiarizarte
4. **📝 Comienza** con el ejercicio básico
5. **🤝 Participa** en el foro para dudas y discusiones

**¡Éxito en tu implementación del Algoritmo Hull-White!** 🎯

---

*Última actualización: 08 de Julio, 2025*