"""
Script para generar los Jupyter Notebooks del curso
Universidad Internacional de La Rioja (UNIR)
Dr. Rodolfo Rafael Medina Ramírez
"""

import json
import os


def crear_notebook_01():
    """Crea el Notebook 01: Introducción a la Frontera Eficiente"""
    
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Introducción a la Frontera Eficiente de Markowitz\n\n",
                    "**Universidad Internacional de La Rioja (UNIR)**  \n",
                    "**Maestría en Ciencias Computacionales y Matemáticas Aplicadas**  \n",
                    "**Profesor:** Dr. Rodolfo Rafael Medina Ramírez\n\n",
                    "---\n\n",
                    "## 🎯 Objetivos\n\n",
                    "1. Comprender los conceptos fundamentales de la teoría moderna de carteras\n",
                    "2. Calcular rendimientos y riesgos de carteras\n",
                    "3. Implementar optimización básica\n",
                    "4. Visualizar la frontera eficiente"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Importar bibliotecas\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from scipy.optimize import minimize\n",
                    "import warnings\n",
                    "warnings.filterwarnings('ignore')\n\n",
                    "plt.style.use('seaborn-v0_8-darkgrid')\n",
                    "%matplotlib inline\n",
                    "pd.options.display.float_format = '{:.4f}'.format\n\n",
                    "print('✅ Bibliotecas importadas')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Datos de Ejemplo\n\n",
                    "Trabajaremos con 5 activos."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Definir activos\n",
                    "activos = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']\n",
                    "rendimientos_esperados = np.array([0.15, 0.12, 0.13, 0.18, 0.20])\n",
                    "volatilidades = np.array([0.25, 0.22, 0.20, 0.30, 0.35])\n\n",
                    "matriz_correlacion = np.array([\n",
                    "    [1.00, 0.75, 0.80, 0.65, 0.55],\n",
                    "    [0.75, 1.00, 0.78, 0.70, 0.50],\n",
                    "    [0.80, 0.78, 1.00, 0.68, 0.52],\n",
                    "    [0.65, 0.70, 0.68, 1.00, 0.60],\n",
                    "    [0.55, 0.50, 0.52, 0.60, 1.00]\n",
                    "])\n\n",
                    "matriz_covarianzas = np.outer(volatilidades, volatilidades) * matriz_correlacion\n",
                    "tasa_libre_riesgo = 0.03\n\n",
                    "print('Datos configurados')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Funciones de Optimización"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def rendimiento_cartera(pesos, rendimientos):\n",
                    "    return np.dot(pesos, rendimientos)\n\n",
                    "def volatilidad_cartera(pesos, cov_matrix):\n",
                    "    return np.sqrt(np.dot(pesos.T, np.dot(cov_matrix, pesos)))\n\n",
                    "def sharpe_ratio(pesos, rendimientos, cov_matrix, rf):\n",
                    "    ret = rendimiento_cartera(pesos, rendimientos)\n",
                    "    vol = volatilidad_cartera(pesos, cov_matrix)\n",
                    "    return (ret - rf) / vol\n\n",
                    "print('✅ Funciones definidas')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def cartera_tangente(rendimientos, cov_matrix, rf):\n",
                    "    n = len(rendimientos)\n",
                    "    \n",
                    "    def objetivo(w):\n",
                    "        return -sharpe_ratio(w, rendimientos, cov_matrix, rf)\n",
                    "    \n",
                    "    restricciones = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}\n",
                    "    limites = tuple((0, 1) for _ in range(n))\n",
                    "    pesos_iniciales = np.array([1/n] * n)\n",
                    "    \n",
                    "    resultado = minimize(objetivo, pesos_iniciales, method='SLSQP',\n",
                    "                        bounds=limites, constraints=restricciones)\n",
                    "    \n",
                    "    pesos = resultado.x\n",
                    "    \n",
                    "    return {\n",
                    "        'pesos': pesos,\n",
                    "        'rendimiento': rendimiento_cartera(pesos, rendimientos),\n",
                    "        'volatilidad': volatilidad_cartera(pesos, cov_matrix),\n",
                    "        'sharpe': sharpe_ratio(pesos, rendimientos, cov_matrix, rf)\n",
                    "    }\n\n",
                    "# Calcular\n",
                    "tangente = cartera_tangente(rendimientos_esperados, matriz_covarianzas, tasa_libre_riesgo)\n\n",
                    "print('⭐ CARTERA TANGENTE')\n",
                    "print(f\"Rendimiento: {tangente['rendimiento']*100:.2f}%\")\n",
                    "print(f\"Volatilidad: {tangente['volatilidad']*100:.2f}%\")\n",
                    "print(f\"Sharpe: {tangente['sharpe']:.4f}\")\n",
                    "print('\\nComposición:')\n",
                    "for i, activo in enumerate(activos):\n",
                    "    if tangente['pesos'][i] > 0.01:\n",
                    "        print(f\"  {activo}: {tangente['pesos'][i]*100:.2f}%\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Visualización\n\n",
                    "Generamos carteras aleatorias para visualizar la frontera."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Generar carteras aleatorias\n",
                    "np.random.seed(42)\n",
                    "n_carteras = 5000\n",
                    "n_activos = len(activos)\n\n",
                    "rendimientos_carteras = []\n",
                    "volatilidades_carteras = []\n",
                    "sharpes_carteras = []\n\n",
                    "for _ in range(n_carteras):\n",
                    "    pesos = np.random.random(n_activos)\n",
                    "    pesos /= np.sum(pesos)\n",
                    "    \n",
                    "    ret = rendimiento_cartera(pesos, rendimientos_esperados)\n",
                    "    vol = volatilidad_cartera(pesos, matriz_covarianzas)\n",
                    "    sr = sharpe_ratio(pesos, rendimientos_esperados, matriz_covarianzas, tasa_libre_riesgo)\n",
                    "    \n",
                    "    rendimientos_carteras.append(ret)\n",
                    "    volatilidades_carteras.append(vol)\n",
                    "    sharpes_carteras.append(sr)\n\n",
                    "rendimientos_carteras = np.array(rendimientos_carteras)\n",
                    "volatilidades_carteras = np.array(volatilidades_carteras)\n",
                    "sharpes_carteras = np.array(sharpes_carteras)\n\n",
                    "print(f'✅ Generadas {n_carteras} carteras')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Gráfico\n",
                    "plt.figure(figsize=(14, 8))\n\n",
                    "scatter = plt.scatter(volatilidades_carteras * 100, rendimientos_carteras * 100,\n",
                    "                     c=sharpes_carteras, cmap='viridis', alpha=0.3, s=10)\n",
                    "plt.colorbar(scatter, label='Ratio de Sharpe')\n\n",
                    "plt.scatter(volatilidades * 100, rendimientos_esperados * 100,\n",
                    "           marker='o', s=150, c='red', edgecolors='black', linewidth=2,\n",
                    "           label='Activos', zorder=3)\n\n",
                    "plt.scatter(tangente['volatilidad'] * 100, tangente['rendimiento'] * 100,\n",
                    "           marker='*', s=500, c='gold', edgecolors='black', linewidth=2,\n",
                    "           label='Tangente', zorder=5)\n\n",
                    "plt.xlabel('Volatilidad (%)', fontsize=12, fontweight='bold')\n",
                    "plt.ylabel('Rendimiento (%)', fontsize=12, fontweight='bold')\n",
                    "plt.title('Frontera Eficiente', fontsize=14, fontweight='bold')\n",
                    "plt.legend()\n",
                    "plt.grid(True, alpha=0.3)\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "---\n\n",
                    "**Universidad Internacional de La Rioja (UNIR)**  \n",
                    "*Última actualización: Agosto 2025*"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.9.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook


def crear_notebook_02():
    """Crea el Notebook 02: Optimización con Datos Reales"""
    
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Optimización de Carteras con Datos Reales\n\n",
                    "**Universidad Internacional de La Rioja (UNIR)**  \n",
                    "**Profesor:** Dr. Rodolfo Rafael Medina Ramírez\n\n",
                    "---\n\n",
                    "## 🎯 Objetivos\n\n",
                    "1. Descargar datos reales con yfinance\n",
                    "2. Optimizar carteras\n",
                    "3. Realizar backtesting\n",
                    "4. Comparar con S&P 500"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "from scipy.optimize import minimize\n",
                    "from datetime import datetime, timedelta\n",
                    "import yfinance as yf\n",
                    "import warnings\n",
                    "warnings.filterwarnings('ignore')\n\n",
                    "print('✅ Bibliotecas importadas')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Configurar activos y periodo\n",
                    "tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'JPM', 'JNJ', 'XOM', 'WMT', 'PG']\n",
                    "fecha_fin = datetime.now()\n",
                    "fecha_inicio = fecha_fin - timedelta(days=365*3)\n\n",
                    "print(f'Descargando {len(tickers)} activos...')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Descargar datos\n",
                    "try:\n",
                    "    datos = yf.download(tickers, start=fecha_inicio, end=fecha_fin, progress=False)\n",
                    "    precios = datos['Adj Close'].dropna()\n",
                    "    print(f'✅ Descargados: {len(precios)} días')\n",
                    "except Exception as e:\n",
                    "    print(f'Error: {e}')\n",
                    "    print('Usando datos de ejemplo...')\n",
                    "    precios = pd.read_csv('../recursos/datasets/precios_historicos_completo.csv', \n",
                    "                          index_col='fecha', parse_dates=True)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Calcular rendimientos\n",
                    "rendimientos = np.log(precios / precios.shift(1)).dropna()\n",
                    "print('Primeros rendimientos:')\n",
                    "print(rendimientos.head())"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Parámetros anualizados\n",
                    "dias_trading = 252\n",
                    "rendimientos_anuales = rendimientos.mean() * dias_trading\n",
                    "volatilidades_anuales = rendimientos.std() * np.sqrt(dias_trading)\n",
                    "matriz_cov = rendimientos.cov() * dias_trading\n\n",
                    "print('Estadísticas calculadas')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Clase Optimizadora"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "class OptimizadorCarteras:\n",
                    "    \n",
                    "    def __init__(self, rendimientos_esp, cov_matrix, rf=0.03):\n",
                    "        self.rendimientos = np.array(rendimientos_esp)\n",
                    "        self.cov_matrix = np.array(cov_matrix)\n",
                    "        self.rf = rf\n",
                    "        self.n_activos = len(rendimientos_esp)\n",
                    "    \n",
                    "    def rendimiento_cartera(self, pesos):\n",
                    "        return np.dot(pesos, self.rendimientos)\n",
                    "    \n",
                    "    def volatilidad_cartera(self, pesos):\n",
                    "        return np.sqrt(np.dot(pesos.T, np.dot(self.cov_matrix, pesos)))\n",
                    "    \n",
                    "    def sharpe_ratio(self, pesos):\n",
                    "        ret = self.rendimiento_cartera(pesos)\n",
                    "        vol = self.volatilidad_cartera(pesos)\n",
                    "        return (ret - self.rf) / vol\n",
                    "    \n",
                    "    def optimizar_tangente(self):\n",
                    "        def objetivo(w):\n",
                    "            return -self.sharpe_ratio(w)\n",
                    "        \n",
                    "        restricciones = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}\n",
                    "        limites = tuple((0, 1) for _ in range(self.n_activos))\n",
                    "        pesos_init = np.array([1/self.n_activos] * self.n_activos)\n",
                    "        \n",
                    "        resultado = minimize(objetivo, pesos_init, method='SLSQP',\n",
                    "                           bounds=limites, constraints=restricciones)\n",
                    "        \n",
                    "        pesos = resultado.x\n",
                    "        \n",
                    "        return {\n",
                    "            'pesos': pesos,\n",
                    "            'rendimiento': self.rendimiento_cartera(pesos),\n",
                    "            'volatilidad': self.volatilidad_cartera(pesos),\n",
                    "            'sharpe': self.sharpe_ratio(pesos)\n",
                    "        }\n\n",
                    "print('✅ Clase definida')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Optimizar\n",
                    "optimizador = OptimizadorCarteras(rendimientos_anuales.values, matriz_cov.values, rf=0.03)\n",
                    "tangente = optimizador.optimizar_tangente()\n\n",
                    "print('⭐ CARTERA TANGENTE')\n",
                    "print(f\"Rendimiento: {tangente['rendimiento']*100:.2f}%\")\n",
                    "print(f\"Volatilidad: {tangente['volatilidad']*100:.2f}%\")\n",
                    "print(f\"Sharpe: {tangente['sharpe']:.4f}\")\n",
                    "print('\\nComposición:')\n",
                    "for i, ticker in enumerate(tickers):\n",
                    "    if tangente['pesos'][i] > 0.01:\n",
                    "        print(f\"  {ticker}: {tangente['pesos'][i]*100:.2f}%\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "---\n\n",
                    "**UNIR - Agosto 2025**  \n",
                    "*Para dudas: Foro Moodle*"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.9.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook


def crear_notebook_03():
    """Crea el Notebook 03: Plantilla para Actividad 02"""
    
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Actividad 02: Construcción de Cartera de Mínimo Riesgo\n",
                    "\n",
                    "**Universidad Internacional de La Rioja (UNIR)**  \n",
                    "**Maestría en Ciencias Computacionales y Matemáticas Aplicadas**  \n",
                    "**Curso:** Modelización y Valoración de Derivados y Carteras en Finanzas  \n",
                    "**Profesor:** Dr. Rodolfo Rafael Medina Ramírez\n",
                    "\n",
                    "---\n",
                    "\n",
                    "## 📋 INFORMACIÓN DEL EQUIPO\n",
                    "\n",
                    "**Integrantes:**\n",
                    "- Nombre 1: [Tu nombre completo]\n",
                    "- Nombre 2: [Nombre completo]\n",
                    "- Nombre 3: [Nombre completo]\n",
                    "- Nombre 4: [Nombre completo]\n",
                    "\n",
                    "**Fecha de entrega:** 18 de agosto de 2025\n",
                    "\n",
                    "---\n",
                    "\n",
                    "## 📝 RESUMEN EJECUTIVO\n",
                    "\n",
                    "*Completa este resumen al final (máximo 250 palabras):*\n",
                    "\n",
                    "[Describe brevemente: (1) Activos seleccionados y justificación, (2) Metodología utilizada, (3) Principales hallazgos, (4) Recomendación final]\n",
                    "\n",
                    "---"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. INTRODUCCIÓN Y OBJETIVOS\n",
                    "\n",
                    "### 1.1 Objetivo General\n",
                    "\n",
                    "*Describe el objetivo principal de este trabajo*\n",
                    "\n",
                    "### 1.2 Objetivos Específicos\n",
                    "\n",
                    "*Lista 3-5 objetivos específicos que quieres lograr*\n",
                    "\n",
                    "1. \n",
                    "2. \n",
                    "3. "
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "---\n",
                    "\n",
                    "## 2. SELECCIÓN Y JUSTIFICACIÓN DE ACTIVOS\n",
                    "\n",
                    "### 2.1 Criterios de Selección\n",
                    "\n",
                    "*Explica los criterios que usaste para seleccionar tus activos*\n",
                    "\n",
                    "### 2.2 Activos Seleccionados\n",
                    "\n",
                    "*Completa la tabla:*\n",
                    "\n",
                    "| Ticker | Nombre | Sector | Justificación |\n",
                    "|--------|--------|--------|---------------|\n",
                    "| XXX    |        |        |               |\n",
                    "| XXX    |        |        |               |\n",
                    "| XXX    |        |        |               |"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "---\n",
                    "\n",
                    "## 3. CONFIGURACIÓN DEL ENTORNO"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Importar bibliotecas necesarias\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from scipy.optimize import minimize\n",
                    "from datetime import datetime, timedelta\n",
                    "import yfinance as yf\n",
                    "import warnings\n",
                    "warnings.filterwarnings('ignore')\n",
                    "\n",
                    "# Configuración de visualización\n",
                    "plt.style.use('seaborn-v0_8-darkgrid')\n",
                    "%matplotlib inline\n",
                    "pd.options.display.float_format = '{:.4f}'.format\n",
                    "\n",
                    "print('✅ Entorno configurado correctamente')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "---\n",
                    "\n",
                    "## 4. DESCARGA Y PREPARACIÓN DE DATOS\n",
                    "\n",
                    "### 4.1 Definir Parámetros"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# TODO: Define tus activos aquí\n",
                    "tickers = ['XXX', 'YYY', 'ZZZ']  # Reemplaza con tus tickers\n",
                    "\n",
                    "# Periodo de análisis (mínimo 1 año, recomendado 2-3 años)\n",
                    "fecha_fin = datetime.now()\n",
                    "fecha_inicio = fecha_fin - timedelta(days=365*2)  # 2 años\n",
                    "\n",
                    "# Tasa libre de riesgo (ajustar según el mercado)\n",
                    "tasa_libre_riesgo = 0.03  # 3% anual\n",
                    "\n",
                    "print(f'Activos seleccionados: {tickers}')\n",
                    "print(f'Periodo: {fecha_inicio.strftime(\"%Y-%m-%d\")} a {fecha_fin.strftime(\"%Y-%m-%d\")}')\n",
                    "print(f'Tasa libre de riesgo: {tasa_libre_riesgo*100:.1f}%')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 4.2 Descargar Datos Históricos"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# TODO: Descarga los datos de tus activos\n",
                    "print('Descargando datos...')\n",
                    "\n",
                    "# COMPLETA ESTE CÓDIGO\n",
                    "# datos = yf.download(...)\n",
                    "# precios = ...\n",
                    "\n",
                    "# Muestra las primeras filas\n",
                    "# print(precios.head())"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 4.3 Calcular Rendimientos"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# TODO: Calcula rendimientos logarítmicos\n",
                    "# rendimientos = ...\n",
                    "\n",
                    "# Muestra estadísticas básicas\n",
                    "# print(rendimientos.describe())"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "---\n",
                    "\n",
                    "## 5. ANÁLISIS EXPLORATORIO DE DATOS\n",
                    "\n",
                    "### 5.1 Evolución de Precios"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# TODO: Grafica la evolución de precios normalizados\n",
                    "# Pista: Normaliza dividiendo por el precio inicial\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "*Análisis:* [Describe qué observas en el gráfico]"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 5.2 Matriz de Correlación"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# TODO: Calcula y visualiza la matriz de correlación\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "*Análisis:* [Interpreta las correlaciones observadas]"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "---\n",
                    "\n",
                    "## 6. CÁLCULO DE PARÁMETROS ESTADÍSTICOS\n",
                    "\n",
                    "### 6.1 Rendimientos y Volatilidades Anualizados"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# TODO: Calcula parámetros anualizados\n",
                    "dias_trading = 252\n",
                    "\n",
                    "# rendimientos_anuales = ...\n",
                    "# volatilidades_anuales = ...\n",
                    "# matriz_cov = ...\n",
                    "\n",
                    "# Crea una tabla resumen\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "---\n",
                    "\n",
                    "## 7. OPTIMIZACIÓN DE CARTERAS\n",
                    "\n",
                    "### 7.1 Implementar Clase Optimizadora"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# TODO: Implementa o importa la clase de optimización\n",
                    "# Puedes usar la clase del Notebook 02 o la de actividad-03/efficient_frontier.py\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 7.2 Cartera de Mínima Varianza"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# TODO: Encuentra la cartera de mínima varianza\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "*Análisis:* [Interpreta los resultados]"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 7.3 Cartera Tangente"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# TODO: Encuentra la cartera tangente (máximo Sharpe)\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "*Análisis:* [Interpreta los resultados]"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 7.4 Visualización de la Frontera Eficiente"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# TODO: Genera y visualiza la frontera eficiente\n",
                    "# Incluye carteras aleatorias para comparación\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "---\n",
                    "\n",
                    "## 8. ANÁLISIS DE SENSIBILIDAD\n",
                    "\n",
                    "### 8.1 Sensibilidad a la Tasa Libre de Riesgo"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# TODO: Analiza cómo cambia la cartera tangente con diferentes Rf\n",
                    "# Prueba valores entre 1% y 5%\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "*Análisis:* [¿Cómo afecta la tasa libre de riesgo a la composición?]"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 8.2 Sensibilidad al Periodo de Estimación"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# TODO: Compara resultados usando diferentes ventanas de tiempo\n",
                    "# Por ejemplo: 1 año vs 2 años vs 3 años\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "---\n",
                    "\n",
                    "## 9. BACKTESTING\n",
                    "\n",
                    "### 9.1 Simulación Histórica"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# TODO: Simula cómo habría performado la cartera en el pasado\n",
                    "# Usa una inversión inicial de $10,000\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 9.2 Métricas de Performance"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# TODO: Calcula:\n",
                    "# - Rendimiento total\n",
                    "# - Rendimiento anualizado\n",
                    "# - Volatilidad realizada\n",
                    "# - Sharpe ratio realizado\n",
                    "# - Máximo drawdown\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "---\n",
                    "\n",
                    "## 10. COMPARACIÓN CON BENCHMARK\n",
                    "\n",
                    "### 10.1 Descargar Datos del Benchmark"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# TODO: Descarga datos del S&P 500 (^GSPC) o un índice apropiado\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 10.2 Comparación de Resultados"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# TODO: Compara tu cartera vs el benchmark\n",
                    "# Crea una tabla comparativa y gráficos\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "*Análisis:* [¿Tu cartera superó al benchmark? ¿Por qué?]"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "---\n",
                    "\n",
                    "## 11. CONCLUSIONES Y RECOMENDACIONES\n",
                    "\n",
                    "### 11.1 Principales Hallazgos\n",
                    "\n",
                    "*Resume los hallazgos más importantes:*\n",
                    "\n",
                    "1. \n",
                    "2. \n",
                    "3. \n",
                    "\n",
                    "### 11.2 Recomendación de Inversión\n",
                    "\n",
                    "*Basado en tu análisis, ¿qué cartera recomendarías?*\n",
                    "\n",
                    "### 11.3 Limitaciones del Análisis\n",
                    "\n",
                    "*¿Qué limitaciones tiene tu estudio?*\n",
                    "\n",
                    "1. \n",
                    "2. \n",
                    "3. \n",
                    "\n",
                    "### 11.4 Trabajo Futuro\n",
                    "\n",
                    "*¿Qué análisis adicionales se podrían realizar?*"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "---\n",
                    "\n",
                    "## 12. REFERENCIAS\n",
                    "\n",
                    "*Lista todas las referencias utilizadas en formato APA:*\n",
                    "\n",
                    "1. Markowitz, H. M. (1952). Portfolio Selection. *The Journal of Finance*, 7(1), 77–91.\n",
                    "2. \n",
                    "3. "
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "---\n",
                    "\n",
                    "## ✅ CHECKLIST DE ENTREGA\n",
                    "\n",
                    "Antes de entregar, verifica que:\n",
                    "\n",
                    "- [ ] Información del equipo completa\n",
                    "- [ ] Resumen ejecutivo (máx 250 palabras)\n",
                    "- [ ] Justificación de selección de activos\n",
                    "- [ ] Todas las celdas ejecutan sin errores\n",
                    "- [ ] Gráficos claros y legibles\n",
                    "- [ ] Análisis e interpretaciones en cada sección\n",
                    "- [ ] Análisis de sensibilidad completo\n",
                    "- [ ] Backtesting implementado\n",
                    "- [ ] Comparación con benchmark\n",
                    "- [ ] Conclusiones y recomendaciones\n",
                    "- [ ] Referencias bibliográficas\n",
                    "- [ ] Código bien comentado\n",
                    "- [ ] Archivo nombrado correctamente: Apellido1_Apellido2_Actividad02.ipynb\n",
                    "\n",
                    "---\n",
                    "\n",
                    "**Universidad Internacional de La Rioja (UNIR)**  \n",
                    "**Fecha límite de entrega:** 18 de agosto de 2025  \n",
                    "*Para dudas, usa el Foro \"Pregúntale a tu profesor\" en Moodle*"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.9.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook


def guardar_notebooks():
    """Guarda los notebooks en archivos .ipynb"""
    
    print("="*70)
    print("GENERADOR DE NOTEBOOKS - UNIR")
    print("Modelización y Valoración de Derivados y Carteras en Finanzas")
    print("="*70)
    print()
    
    # Notebook 01
    print("📝 Creando Notebook 01: Introducción a la Frontera Eficiente...")
    nb01 = crear_notebook_01()
    with open('01_introduccion_frontera_eficiente.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb01, f, indent=1, ensure_ascii=False)
    print("   ✅ Completado")
    
    # Notebook 02
    print("\n📝 Creando Notebook 02: Optimización con Datos Reales...")
    nb02 = crear_notebook_02()
    with open('02_optimizacion_datos_reales.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb02, f, indent=1, ensure_ascii=False)
    print("   ✅ Completado")
    
    # Notebook 03
    print("\n📝 Creando Notebook 03: Plantilla para Actividad 02...")
    nb03 = crear_notebook_03()
    with open('03_plantilla_actividad_02.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb03, f, indent=1, ensure_ascii=False)
    print("   ✅ Completado")
    
    print("\n" + "="*70)
    print("✅ TODOS LOS NOTEBOOKS CREADOS EXITOSAMENTE")
    print("="*70)
    print("\n📁 Archivos generados:")
    print("   1. 01_introduccion_frontera_eficiente.ipynb")
    print("   2. 02_optimizacion_datos_reales.ipynb")
    print("   3. 03_plantilla_actividad_02.ipynb")
    print("\n💡 Ahora puedes:")
    print("   - Abrirlos en VS Code")
    print("   - Abrirlos en Jupyter Notebook")
    print("   - Subirlos a Google Colab")
    print("\n🎓 ¡Listo para compartir con tus estudiantes!")
    print("="*70)


if __name__ == "__main__":
    guardar_notebooks()