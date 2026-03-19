# Changelog

Todos los cambios notables en este proyecto serán documentados en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/),
y este proyecto adhiere a [Versionado Semántico](https://semver.org/lang/es/).

---

## [2.0.0] - 2025-08-05

### 🎉 Lanzamiento Mayor - Módulo de Optimización de Carteras

Esta versión introduce un módulo completo de optimización de carteras financieras con notebooks educativos interactivos.

### ✨ Añadido

#### Módulo de Optimización (`actividad-03/`)
- **`efficient_frontier.py`**: Clase completa para optimización de carteras
  - Cartera de mínima varianza
  - Cartera tangente (máximo Sharpe)
  - Cartera para rendimiento objetivo
  - Generación de frontera eficiente
  - Visualización con matplotlib
- **`requirements.txt`**: Dependencias del módulo
- **`README.md`**: Documentación completa con ejemplos de uso

#### Notebooks Educativos (`notebooks/`)
- **`01_introduccion_frontera_eficiente.ipynb`**: 
  - Conceptos fundamentales de Markowitz
  - Ejemplos con datos sintéticos (5 activos)
  - Optimización básica
  - Visualización de frontera eficiente
  - Ejercicios prácticos para estudiantes
  - Duración estimada: 60-90 minutos

- **`02_optimizacion_datos_reales.ipynb`**:
  - Descarga de datos con yfinance
  - Análisis exploratorio de datos
  - Optimización con 10 activos reales
  - Backtesting histórico
  - Comparación con S&P 500
  - Cálculo de alpha y beta
  - Duración estimada: 90-120 minutos

- **`03_plantilla_actividad_02.ipynb`**:
  - Plantilla estructurada para la Actividad 02
  - 12 secciones guiadas con instrucciones
  - Espacios para análisis e interpretación
  - Checklist completo de entrega
  - Formato profesional listo para entregar

- **`README.md`**: Guía completa de uso de notebooks
  - Instrucciones de instalación (Colab y local)
  - Descripción detallada de cada notebook
  - Ruta de aprendizaje sugerida
  - Solución de problemas comunes
  - FAQ completo
  - Rúbrica de evaluación

- **`crear_notebooks.py`**: Script Python para generar los notebooks automáticamente

#### Recursos y Datasets (`recursos/datasets/`)
- **`generar_datos.py`**: Generador de datos sintéticos
  - 10 activos con correlaciones realistas
  - 252 días de trading
  - Parámetros configurables
  
- **`activos_ejemplo.csv`**: Información de 10 activos de ejemplo
- **`matriz_correlacion.csv`**: Matriz de correlación predefinida
- **`README.md`**: Documentación de datasets con ejemplos de uso

#### Documentación
- **`CHANGELOG.md`**: Este archivo (control de versiones)
- **README.md actualizado**: 
  - Nueva sección de Actividad 03
  - Tabla de actividades actualizada
  - Estructura de carpetas completa
  - Inicio rápido mejorado
  - Versión 2.0

### 🔄 Cambiado

#### Política de Comunicación
- **IMPORTANTE**: Eliminadas TODAS las referencias a correo electrónico
- Canal oficial: **SOLO Foro "Pregúntale a tu profesor" en Moodle**
- Actualizado en todos los README y documentación
- Advertencia explícita: no se responderán consultas por otros medios

#### Actividad 02 del Ciclo Actual
- **Nueva fecha de entrega**: 18 de agosto de 2025
- **Tema actualizado**: "Construcción de una Cartera Financiera de Mínimo Riesgo"
- Soporte completo con 3 notebooks interactivos
- Datasets de ejemplo incluidos

### 📚 Dependencias Nuevas
```txt
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.5.0
seaborn>=0.11.0
yfinance>=0.2.0
jupyter>=1.0.0
notebook>=6.4.0
ipywidgets>=7.6.0
```

### 🎓 Para Estudiantes

#### Nuevos Recursos Disponibles
1. **Notebooks interactivos**: Aprendizaje paso a paso con código ejecutable
2. **Datos de ejemplo**: No requiere conexión a internet para practicar
3. **Plantilla de actividad**: Estructura clara para la entrega
4. **Documentación extensa**: Guías, FAQs, y solución de problemas

#### Ruta de Aprendizaje Recomendada
1. Notebook 01 → Fundamentos (60-90 min)
2. Notebook 02 → Datos reales (90-120 min)
3. Notebook 03 → Completar actividad
4. Revisión con checklist incluido

### 🔧 Mejoras Técnicas

- Optimización con `scipy.optimize.minimize`
- Método SLSQP para restricciones
- Soporte para pandas y numpy arrays
- Manejo de errores mejorado
- Visualizaciones profesionales con seaborn

### 📊 Métricas del Proyecto

- **Archivos nuevos**: 15+
- **Líneas de código Python**: ~800
- **Notebooks**: 3 completos
- **Datasets**: 6 archivos CSV
- **Documentación**: 5 README actualizados

---

## [1.0.0] - 2025-07-15

### ✨ Versión Inicial

#### Añadido
- **Actividad 01**: Análisis de estrategia Long Straddle
  - Cálculo de payoff
  - Análisis de breakeven
  - Visualización de perfiles
  
- **Actividad 02**: Algoritmo Hull-White
  - Implementación en Python
  - Implementación en MATLAB
  - Calibración de parámetros
  
- **Estructura básica del repositorio**
- **README.md inicial**
- **Licencia y documentación básica**

---

## Tipos de Cambios

- **✨ Añadido**: para funcionalidades nuevas
- **🔄 Cambiado**: para cambios en funcionalidades existentes
- **🗑️ Obsoleto**: para funcionalidades que pronto se eliminarán
- **❌ Eliminado**: para funcionalidades eliminadas
- **🐛 Corregido**: para corrección de errores
- **🔒 Seguridad**: en caso de vulnerabilidades

---

## Enlaces

- [Repositorio GitHub](https://github.com/prof-rodolfo-medina/modeladoCarterasFinanzas-202502)
- [Aula Virtual Moodle](https://aulavirtual.unir.net) - Para dudas y consultas
- [Documentación de Notebooks](./notebooks/README.md)
- [Guía de Actividad 03](./actividad-03/README.md)

---

**Universidad Internacional de La Rioja (UNIR)**  
**Dr. Rodolfo Rafael Medina Ramírez**  
*Última actualización: Agosto 2025*