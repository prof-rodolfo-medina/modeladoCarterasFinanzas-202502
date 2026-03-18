# Jupyter Notebooks - Optimización de Carteras Financieras

**Universidad Internacional de La Rioja (UNIR)**  
**Maestría en Ciencias Computacionales y Matemáticas Aplicadas**  
**Curso:** Modelización y Valoración de Derivados y Carteras en Finanzas  
**Profesor:** Dr. Rodolfo Rafael Medina Ramírez

---

## 📚 Contenido de la Carpeta

Esta carpeta contiene notebooks interactivos de Jupyter para aprender optimización de carteras financieras y completar la **Actividad 02: Construcción de una Cartera Financiera de Mínimo Riesgo**.

### 📁 Notebooks Disponibles

| Notebook | Descripción | Nivel | Duración |
|----------|-------------|-------|----------|
| **01_introduccion_frontera_eficiente.ipynb** | Conceptos fundamentales, ejemplos simples, visualización básica | Principiante | 60-90 min |
| **02_optimizacion_datos_reales.ipynb** | Descarga de datos reales, optimización avanzada, backtesting | Intermedio | 90-120 min |
| **03_plantilla_actividad_02.ipynb** | Plantilla guiada para completar la Actividad 02 | Todos | Variable |

---

## 🚀 Inicio Rápido

### Opción 1: Google Colab (Recomendado - No requiere instalación)

1. **Abrir en Google Colab:**
   - Ve a [Google Colab](https://colab.research.google.com/)
   - Click en `Archivo → Abrir notebook`
   - Selecciona la pestaña `GitHub`
   - Pega la URL del repositorio: `https://github.com/prof-rodolfo-medina/modeladoCarterasFinanzas-202502`
   - Selecciona el notebook que deseas abrir

2. **Ejecutar:**
   - Las celdas se ejecutan con `Shift + Enter`
   - Todas las dependencias se instalan automáticamente
   - Los datos se descargan directamente desde Yahoo Finance

3. **Guardar tu trabajo:**
   - `Archivo → Guardar una copia en Drive`
   - Esto crea tu propia copia editable

### Opción 2: Instalación Local

#### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Jupyter Notebook o JupyterLab

#### Instalación
```bash
# 1. Clonar el repositorio
git clone https://github.com/prof-rodolfo-medina/modeladoCarterasFinanzas-202502.git
cd modeladoCarterasFinanzas-202502/notebooks

# 2. Crear entorno virtual (recomendado)
python -m venv venv

# Activar entorno (Windows)
venv\Scripts\activate

# Activar entorno (Linux/Mac)
source venv/bin/activate

# 3. Instalar dependencias
pip install -r ../actividad-03/requirements.txt

# 4. Instalar Jupyter
pip install jupyter notebook

# 5. Iniciar Jupyter Notebook
jupyter notebook
```

#### Dependencias Principales
```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scipy>=1.7.0
yfinance>=0.2.0
```

---

## 📖 Guía de Uso por Notebook

### 📘 Notebook 01: Introducción a la Frontera Eficiente

**¿Qué aprenderás?**

- ✅ Conceptos fundamentales de la teoría de Markowitz
- ✅ Cálculo de rendimientos y riesgos de carteras
- ✅ Optimización básica (mínima varianza y tangente)
- ✅ Visualización de la frontera eficiente
- ✅ Interpretación del ratio de Sharpe

**¿Cuándo usarlo?**

- Primera vez trabajando con optimización de carteras
- Necesitas repasar conceptos teóricos
- Quieres entender los fundamentos antes de datos reales

**Estructura:**

1. Configuración del entorno
2. Conceptos fundamentales (teoría)
3. Datos de ejemplo (5 activos sintéticos)
4. Cálculo de métricas básicas
5. Optimización simple
6. Visualización de resultados
7. Ejercicios prácticos

**Ejercicios Incluidos:**

- ✏️ Cartera con restricciones (max 30% por activo)
- ✏️ Análisis de sensibilidad a Rf
- ✏️ Cartera para rendimiento objetivo

---

### 📗 Notebook 02: Optimización con Datos Reales

**¿Qué aprenderás?**

- ✅ Descargar datos de Yahoo Finance con `yfinance`
- ✅ Procesar y limpiar datos históricos
- ✅ Calcular parámetros estadísticos robustos
- ✅ Implementar optimización avanzada
- ✅ Realizar backtesting completo
- ✅ Comparar con benchmarks (S&P 500)
- ✅ Calcular alpha de Jensen y beta

**¿Cuándo usarlo?**

- Ya entiendes los conceptos básicos
- Quieres trabajar con datos reales del mercado
- Necesitas implementar análisis completo
- Preparación para la Actividad 02

**Estructura:**

1. Descarga de datos reales (10 acciones, 3 años)
2. Análisis exploratorio de datos
3. Cálculo de parámetros estadísticos
4. Optimización de carteras (múltiples métodos)
5. Visualización avanzada (frontera eficiente completa)
6. Análisis de sensibilidad
   - Sensibilidad a tasa libre de riesgo
   - Sensibilidad al periodo de estimación
7. Backtesting histórico
   - Evolución del valor
   - Rendimientos mensuales
   - Drawdown analysis
8. Comparación con S&P 500
   - Métricas de performance
   - Cálculo de alpha y beta
   - Análisis de correlación

**Activos Incluidos:**

- **Tecnología:** AAPL, GOOGL, MSFT
- **Consumo:** AMZN, TSLA, WMT, PG
- **Financiero:** JPM
- **Salud:** JNJ
- **Energía:** XOM

**Métricas Calculadas:**

- Rendimiento total y anualizado
- Volatilidad anualizada
- Ratio de Sharpe
- Máximo drawdown
- Alpha de Jensen
- Beta de mercado
- R² (coeficiente de determinación)

---

### 📙 Notebook 03: Plantilla para Actividad 02

**¿Qué contiene?**

- ✅ Estructura completa para la actividad
- ✅ Secciones pre-formateadas
- ✅ Instrucciones paso a paso
- ✅ Espacios para tu código
- ✅ Checklist de entregables
- ✅ Rúbrica de evaluación

**¿Cuándo usarlo?**

- Estás listo para completar la Actividad 02
- Necesitas una estructura clara para tu trabajo
- Quieres asegurar que incluyes todo lo requerido

**Secciones Incluidas:**

1. **Portada y Resumen Ejecutivo**
2. **Introducción y Objetivos**
3. **Selección y Justificación de Activos**
4. **Metodología**
5. **Análisis Exploratorio de Datos**
6. **Optimización de Carteras**
7. **Análisis de Resultados**
8. **Backtesting**
9. **Conclusiones y Recomendaciones**
10. **Referencias**

---

## 🎯 Ruta de Aprendizaje Sugerida

### Para Principiantes
```
Día 1-2: Notebook 01 (Introducción)
├── Leer toda la teoría
├── Ejecutar todas las celdas
├── Entender cada visualización
└── Completar ejercicios básicos

Día 3-4: Notebook 02 (Primera parte)
├── Secciones 1-4: Descarga y optimización
├── Practicar con diferentes activos
└── Experimentar con parámetros

Día 5-6: Notebook 02 (Segunda parte)
├── Secciones 5-8: Análisis avanzado
├── Interpretar resultados
└── Documentar hallazgos

Día 7+: Notebook 03 (Actividad 02)
├── Aplicar todo lo aprendido
├── Trabajo en equipo
└── Documentación profesional
```

### Para Estudiantes con Experiencia
```
Día 1: Revisión rápida de Notebook 01
└── Enfocarse en ejercicios avanzados

Día 2-3: Notebook 02 completo
├── Modificar código para tus activos
└── Agregar análisis adicionales

Día 4+: Notebook 03 (Actividad 02)
├── Personalizar metodología
├── Agregar análisis innovadores
└── Documentación excepcional
```

---

## 💡 Consejos y Mejores Prácticas

### Antes de Empezar

- [ ] Lee el README completo del repositorio
- [ ] Revisa los objetivos de la Actividad 02
- [ ] Asegúrate de tener todas las dependencias instaladas
- [ ] Ten a mano las presentaciones del curso

### Durante el Trabajo

- [ ] **Ejecuta las celdas en orden** (de arriba hacia abajo)
- [ ] **No borres las salidas** al guardar (ayudan a revisar)
- [ ] **Comenta tu código** para recordar tu lógica
- [ ] **Guarda frecuentemente** (Ctrl+S / Cmd+S)
- [ ] **Experimenta** modificando parámetros
- [ ] **Documenta** tus observaciones

### Al Finalizar

- [ ] Verifica que todas las celdas ejecutan sin errores
- [ ] Revisa que todas las visualizaciones se muestran correctamente
- [ ] Lee tus conclusiones para verificar coherencia
- [ ] Exporta a PDF si es requerido (`Archivo → Descargar como → PDF`)

---

## 🔧 Solución de Problemas Comunes

### Error: "ModuleNotFoundError: No module named 'yfinance'"

**Solución:**
```bash
pip install yfinance
```

**En Google Colab:**
```python
!pip install yfinance
```

### Error: "No data found, symbol may be delisted"

**Causa:** El símbolo de la acción no existe o cambió.

**Solución:**
- Verifica el ticker en Yahoo Finance
- Usa símbolos válidos (ej: AAPL, MSFT)
- Asegúrate de tener conexión a internet

### Error: "DataFrame object has no attribute 'append'"

**Causa:** pandas >= 2.0 eliminó el método `.append()`

**Solución:**
```python
# Antiguo (no funciona en pandas 2.0+)
df = df.append(new_data)

# Nuevo (funciona en todas las versiones)
df = pd.concat([df, new_data], ignore_index=True)
```

### Las visualizaciones no se muestran

**Solución:**
```python
# Añadir al inicio del notebook
%matplotlib inline

# O para gráficos interactivos
%matplotlib widget
```

### Error: "KeyError" al acceder a columnas

**Causa:** El nombre de la columna no existe.

**Solución:**
```python
# Ver nombres de columnas
print(df.columns.tolist())

# Acceder correctamente
df['nombre_correcto']
```

### El notebook está muy lento

**Soluciones:**
- Reduce el número de carteras aleatorias (de 10000 a 5000)
- Usa menos puntos en la frontera eficiente (de 100 a 50)
- Cierra otros programas
- En Colab: Usa GPU (`Runtime → Change runtime type → GPU`)

---

## 📊 Datos y Recursos Adicionales

### Fuentes de Datos

1. **Yahoo Finance** (usado en los notebooks)
   - Gratuito, fácil de usar
   - Datos históricos de acciones
   - API: `yfinance`

2. **Alternativas** (para proyectos avanzados)
   - Alpha Vantage: https://www.alphavantage.co/
   - Quandl: https://www.quandl.com/
   - FRED (datos económicos): https://fred.stlouisfed.org/

### Datasets de Ejemplo

En la carpeta `/recursos/datasets/` encontrarás:
- `activos_ejemplo.csv` - Información de 10 activos
- `rendimientos_historicos.csv` - 252 días de rendimientos
- `matriz_correlacion.csv` - Matriz de correlación
- `ejemplo_simple_3_activos.csv` - Dataset simplificado

### Documentación Útil

- **NumPy:** https://numpy.org/doc/
- **Pandas:** https://pandas.pydata.org/docs/
- **Matplotlib:** https://matplotlib.org/stable/contents.html
- **SciPy Optimize:** https://docs.scipy.org/doc/scipy/reference/optimize.html
- **yfinance:** https://pypi.org/project/yfinance/

---

## 🤝 Trabajo en Equipo (Actividad 02)

### Organización Sugerida

**Opción 1: División por Tareas**
- **Miembro 1:** Descarga de datos y análisis exploratorio
- **Miembro 2:** Optimización y cálculos
- **Miembro 3:** Visualizaciones y backtesting
- **Miembro 4:** Documentación y conclusiones

**Opción 2: División por Activos**
- Cada miembro investiga un grupo de activos
- Todos participan en la optimización conjunta
- Reuniones para integrar resultados

### Herramientas de Colaboración

1. **Google Colab + Drive**
   - Todos editan el mismo notebook
   - Cambios se sincronizan automáticamente
   - Comentarios en celdas específicas

2. **GitHub**
   - Clonar repositorio
   - Cada uno trabaja en branch
   - Merge final antes de entregar

3. **Jupyter on the Cloud**
   - JupyterHub compartido
   - Acceso simultáneo
   - Control de versiones

---

## 📝 Entrega de la Actividad 02

### Formatos Aceptados

1. **Jupyter Notebook (.ipynb)**
   - Todas las celdas ejecutadas
   - Salidas visibles
   - Bien documentado

2. **PDF exportado desde Jupyter**
   - `File → Download as → PDF via LaTeX`
   - O `File → Print Preview → Save as PDF`

3. **HTML exportado**
   - `File → Download as → HTML`
   - Incluye todas las visualizaciones

### Checklist de Entrega

- [ ] Portada con datos del equipo
- [ ] Resumen ejecutivo (max 1 página)
- [ ] Justificación de selección de activos
- [ ] Código bien comentado
- [ ] Visualizaciones claras y profesionales
- [ ] Análisis de sensibilidad incluido
- [ ] Backtesting completo
- [ ] Conclusiones y recomendaciones
- [ ] Referencias bibliográficas
- [ ] Todos los archivos nombrados correctamente

### Nomenclatura de Archivos
```
Apellido1_Apellido2_Actividad02.ipynb
Apellido1_Apellido2_Actividad02.pdf
datos_Apellido1_Apellido2.csv  (si aplica)
```

---

## 🆘 Soporte y Ayuda

### Canal Oficial de Comunicación

**⚠️ IMPORTANTE:** Por política institucional, la **ÚNICA vía oficial de comunicación** con el profesor es:

📢 **Foro: "Pregúntale a tu profesor"** en el Aula Virtual Moodle

- ✅ Publica tu pregunta en el foro
- ✅ Respuesta garantizada en máximo 48 horas hábiles
- ✅ Tus compañeros también pueden ayudar
- ✅ Las respuestas quedan disponibles para todos

### Cómo Hacer una Buena Pregunta en el Foro

1. **Título descriptivo:** "Error al descargar datos en Notebook 02"
2. **Contexto:** Qué estabas intentando hacer
3. **Código relevante:** Copia la celda que da error
4. **Mensaje de error:** Copia el error completo
5. **Qué has intentado:** Soluciones que ya probaste

**Ejemplo de buena pregunta:**
```
Título: Error ModuleNotFoundError en Notebook 02

Hola profesor,

Estoy trabajando en el Notebook 02 y al ejecutar la celda de 
importación de yfinance obtengo el siguiente error:

ModuleNotFoundError: No module named 'yfinance'

Ya intenté ejecutar:
pip install yfinance

Pero sigue sin funcionar. ¿Qué más puedo hacer?

Gracias.
```

### Otros Recursos de Ayuda

1. **Documentación del Repositorio**
   - README principal
   - QUICK_START.md
   - Comentarios en el código

2. **Compañeros de Clase**
   - Foro de estudiantes en Moodle
   - Grupos de estudio
   - Trabajo colaborativo

3. **Recursos en Línea**
   - Stack Overflow (para errores técnicos de Python)
   - Documentación oficial de bibliotecas
   - Tutoriales en YouTube

### Preguntas Frecuentes (FAQ)

**P: ¿Puedo usar otros activos además de los del ejemplo?**  
R: ¡Absolutamente! De hecho, se recomienda. Debes justificar tu selección.

**P: ¿Cuántos activos debo incluir?**  
R: Mínimo 3, recomendado 5-10 para buena diversificación.

**P: ¿Puedo trabajar solo en la Actividad 02?**  
R: Sí, aunque se recomienda trabajo en equipo (2-4 personas).

**P: ¿Qué periodo de datos debo usar?**  
R: Mínimo 1 año, recomendado 2-3 años de datos históricos.

**P: ¿Es obligatorio usar Python?**  
R: Puedes usar Python, R, MATLAB o Excel. Python es recomendado.

**P: ¿Qué pasa si Yahoo Finance no funciona?**  
R: Usa los datasets de ejemplo en `/recursos/datasets/`

**P: ¿Dónde pregunto si tengo dudas sobre el código?**  
R: En el foro "Pregúntale a tu profesor" del Aula Virtual Moodle.

---

## 📚 Recursos Adicionales

### Lecturas Recomendadas

- Markowitz, H. M. (1952). "Portfolio Selection". *The Journal of Finance*.
- Sharpe, W. F. (1964). "Capital Asset Prices". *The Journal of Finance*.
- Bodie, Kane & Marcus (2018). "Investments" (Capítulos 6-9).

### Videos Tutoriales

- **Python para Finanzas:** [YouTube Playlist]
- **Optimización de Carteras:** [Coursera]
- **yfinance Tutorial:** [Medium Articles]

### Bibliotecas Especializadas
```python
# Para optimización avanzada
pip install PyPortfolioOpt

# Para análisis financiero
pip install pandas-datareader

# Para visualización interactiva
pip install plotly
```

---

## 🎓 Evaluación y Criterios

### Rúbrica Simplificada

| Criterio | Peso | Descripción |
|----------|------|-------------|
| **Código** | 30% | Funciona correctamente, bien comentado |
| **Análisis** | 40% | Completo, correcto, bien interpretado |
| **Visualización** | 15% | Clara, profesional, informativa |
| **Documentación** | 15% | Bien estructurada, sin errores |

### Niveles de Desempeño

- **Excelente (9-10):** Análisis profundo, código impecable, insights originales
- **Bueno (7-8):** Cumple todos los requisitos, análisis correcto
- **Satisfactorio (6-7):** Cumple requisitos mínimos, algunos errores menores
- **Insuficiente (<6):** Requisitos incompletos o errores graves

---

## 🔄 Actualizaciones

**Última actualización:** Agosto 2025

**Historial de cambios:**
- v1.0 (Ago 2025): Versión inicial con 3 notebooks

**Próximas mejoras:**
- Notebook 04: Modelos de riesgo (VaR, CVaR)
- Notebook 05: Black-Litterman
- Notebook 06: Rebalanceo dinámico

---

## 📧 Contacto

**Profesor:** Dr. Rodolfo Rafael Medina Ramírez  
**Canal de comunicación oficial:** Foro "Pregúntale a tu profesor" en Aula Virtual Moodle  
**Repositorio GitHub:** https://github.com/prof-rodolfo-medina/modeladoCarterasFinanzas-202502

⚠️ **IMPORTANTE:** No se responderán consultas por otros medios. Toda comunicación debe realizarse a través del foro oficial del curso.

---

## 📜 Licencia

Este material es de uso exclusivo para estudiantes del curso **Modelización y Valoración de Derivados y Carteras en Finanzas** de UNIR México. 

**Prohibido:**
- ❌ Distribución comercial
- ❌ Uso fuera del curso
- ❌ Modificación sin autorización

**Permitido:**
- ✅ Uso personal para aprendizaje
- ✅ Compartir con compañeros del curso
- ✅ Modificar para tareas del curso

---

**¡Éxito en tu aprendizaje! 🚀**

*Si encuentras errores o tienes sugerencias, por favor repórtalos en el foro "Pregúntale a tu profesor" del Aula Virtual Moodle.*