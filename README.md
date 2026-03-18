# Modelado y Valoración de Carteras en Finanzas 2025-02

**Universidad Internacional de La Rioja (UNIR)**  
**Maestría en Ciencias Computacionales y Matemáticas Aplicadas**  
**Profesor:** Dr. Rodolfo Rafael Medina Ramírez

---

## 🎯 Objetivo del Curso

Aprender análisis cuantitativo aplicado a finanzas mediante programación en Python, con énfasis en optimización de carteras, gestión de riesgos y modelado financiero.

---

## 📚 Contenido del Repositorio

### 📁 **Actividades**

#### [Actividad 01](./actividad-01/) - Análisis de Estrategia Long Straddle
- Análisis de estrategias de opciones
- Cálculo de payoff y breakeven
- Visualización de perfiles riesgo-rendimiento

#### [Actividad 02](./actividad-02/) - Algoritmo Hull-White
- Calibración de parámetros binomiales
- Implementación en Python y MATLAB
- Comparación de resultados

#### [Actividad 03](./actividad-03/) - **NUEVO** Optimización de Carteras
- Frontera eficiente de Markowitz
- Cartera de mínimo riesgo
- Cartera tangente (máximo Sharpe)
- Análisis de sensibilidad

---

### 📓 **Notebooks Jupyter**

Notebooks interactivos para aprendizaje práctico:

1. **[01_introduccion_frontera_eficiente.ipynb](./notebooks/)** - Conceptos básicos
2. **[02_optimizacion_datos_reales.ipynb](./notebooks/)** - Implementación con datos reales
3. **[03_plantilla_actividad_02.ipynb](./notebooks/)** - Plantilla para la actividad

📖 [Ver guía completa de notebooks](./notebooks/README.md)

---

### 📊 **Recursos**

#### [Datasets](./recursos/datasets/)
- Datos de ejemplo para práctica
- Script generador de datos sintéticos
- Datos históricos de activos financieros

---

## 🚀 Inicio Rápido

### Opción 1: Google Colab (Recomendado)
```
1. Ve a https://colab.research.google.com/
2. Archivo → Abrir notebook → GitHub
3. Pega: https://github.com/prof-rodolfo-medina/modeladoCarterasFinanzas-202502
4. Selecciona el notebook que deseas abrir
```

### Opción 2: Instalación Local
```bash
# 1. Clonar repositorio
git clone https://github.com/prof-rodolfo-medina/modeladoCarterasFinanzas-202502.git
cd modeladoCarterasFinanzas-202502

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows

# 3. Instalar dependencias
pip install -r actividad-03/requirements.txt

# 4. Ejecutar
python actividad-03/efficient_frontier.py
# o
jupyter notebook notebooks/
```

---

## 📋 Actividades y Fechas de Entrega

| Actividad | Descripción | Fecha Publicación | Fecha Entrega | Estado |
|-----------|-------------|-------------------|---------------|--------|
| Actividad 01 | Long Straddle | 26 Jun 2025 | 01 Jul 2025 | ✅ Completada |
| Actividad 02 | Hull-White | 08 Jul 2025 | 15 Jul 2025 | ✅ Completada |
| **Actividad 02** | **Cartera Mínimo Riesgo** | **05 Ago 2025** | **18 Ago 2025** | 🔥 **ACTIVA** |

---

## 🛠️ Herramientas Requeridas

### Esenciales
- **Python 3.8+**
- **Jupyter Notebook** (o Google Colab)

### Bibliotecas Principales
```bash
pip install numpy pandas matplotlib scipy yfinance
```

### Opcionales
- **MATLAB R2020a+** (para comparaciones)
- **Visual Studio Code** (IDE recomendado)

---

## 📖 Estructura de Carpetas
```
modeladoCarterasFinanzas-202502/
├── actividad-01/          # Long Straddle
├── actividad-02/          # Hull-White
├── actividad-03/          # Optimización de Carteras (NUEVO)
│   ├── efficient_frontier.py
│   ├── requirements.txt
│   └── README.md
├── notebooks/             # Jupyter Notebooks (NUEVO)
│   ├── 01_introduccion_frontera_eficiente.ipynb
│   ├── 02_optimizacion_datos_reales.ipynb
│   ├── 03_plantilla_actividad_02.ipynb
│   └── README.md
├── recursos/
│   └── datasets/          # Datos de ejemplo (NUEVO)
│       ├── generar_datos.py
│       ├── activos_ejemplo.csv
│       ├── matriz_correlacion.csv
│       └── README.md
└── README.md              # Este archivo
```

---

## 💡 Cómo Usar Este Repositorio

### Para Estudiantes:

1. **Explorar notebooks** - Comienza con `notebooks/01_introduccion_frontera_eficiente.ipynb`
2. **Practicar con ejemplos** - Usa los datasets en `recursos/datasets/`
3. **Completar actividades** - Sigue las instrucciones en cada carpeta `actividad-XX/`
4. **Preguntar dudas** - Usa el Foro "Pregúntale a tu profesor" en Moodle

### Para Mantenerse Actualizado:
```bash
git pull origin main
```

---

## 🆘 Soporte

### Canal Oficial de Comunicación

⚠️ **IMPORTANTE:** Por política institucional, la **ÚNICA vía oficial** de comunicación es:

📢 **Foro: "Pregúntale a tu profesor"** en el Aula Virtual Moodle

- ✅ Respuesta garantizada en máximo 48 horas hábiles
- ✅ Las respuestas quedan disponibles para todos
- ✅ Tus compañeros también pueden ayudar

### Recursos Adicionales

- 📚 [Documentación de NumPy](https://numpy.org/doc/)
- 📊 [Documentación de Pandas](https://pandas.pydata.org/docs/)
- 📈 [Documentación de Matplotlib](https://matplotlib.org/)
- 🔧 [SciPy Optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html)

---

## 📚 Referencias Bibliográficas

- Markowitz, H. M. (1952). "Portfolio Selection". *The Journal of Finance*, 7(1), 77–91.
- Sharpe, W. F. (1964). "Capital Asset Prices". *The Journal of Finance*, 19(3), 425–442.
- Bodie, Z., Kane, A., & Marcus, A. J. (2018). *Investments* (11th ed.). McGraw-Hill.
- Elton, E. J., et al. (2014). *Modern Portfolio Theory and Investment Analysis* (9th ed.). Wiley.

---

## 🔄 Actualizaciones Recientes

### v2.0 (Agosto 2025) - 🆕 NUEVO
- ✅ Añadido módulo de optimización de carteras (`actividad-03/`)
- ✅ Creados 3 Jupyter Notebooks interactivos
- ✅ Generador de datasets sintéticos
- ✅ Documentación completa actualizada

### v1.0 (Julio 2025)
- Actividades 01 y 02 iniciales
- Estructura base del repositorio

---

## 📝 Notas para Estudiantes

⚠️ **Política de Integridad Académica:**
- El código es de apoyo para el aprendizaje
- Debes entender cada línea que entregas
- El trabajo debe ser original de tu equipo
- Cita apropiadamente cualquier código externo

✅ **Mejores Prácticas:**
- Comenta tu código
- Usa nombres de variables descriptivos
- Documenta tus resultados
- Haz commits frecuentes si usas Git

---

## 📜 Licencia

Este material es de uso exclusivo para estudiantes del curso **Modelización y Valoración de Derivados y Carteras en Finanzas** de UNIR México.

**Prohibido:**
- ❌ Distribución comercial
- ❌ Uso fuera del curso
- ❌ Plagio académico

**Permitido:**
- ✅ Uso personal para aprendizaje
- ✅ Compartir con compañeros del curso
- ✅ Modificar para tareas del curso

---

## 🌟 Contribuciones

Si encuentras errores o tienes sugerencias de mejora, repórtalos en el **Foro del Aula Virtual Moodle**.

---

**¡Éxito en tu aprendizaje! 🚀**

*Universidad Internacional de La Rioja - UNIR México*  
*Última actualización: Agosto 2025*