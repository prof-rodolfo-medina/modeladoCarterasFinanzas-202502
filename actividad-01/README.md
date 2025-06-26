# 📈 Actividad 01 - Análisis de Estrategia Long Straddle

## 🎯 Objetivos
Analizar una estrategia de opciones conocida como **Long Straddle**, que consiste en la compra simultánea de una opción call y una opción put con el mismo precio de ejercicio y fecha de vencimiento.

## 📊 Parámetros del Ejercicio
- **Prima Call (C):** 5€
- **Prima Put (P):** 4€  
- **Precio de Ejercicio (E):** 100€
- **Desembolso inicial:** 9€

## 📁 Archivos Incluidos

### `long_straddle.py`
Código principal que incluye:
- Cálculo de payoffs individuales (call y put)
- Análisis de puntos de equilibrio
- Visualización gráfica del perfil de ganancia/pérdida
- Ejemplos numéricos específicos

## 🚀 Cómo Ejecutar

### Prerrequisitos:
```bash
pip install numpy matplotlib
```

### Ejecución:
```bash
python long_straddle.py
```

## 📈 Resultados Esperados

### Puntos Clave:
- **Breakeven inferior:** 91€
- **Breakeven superior:** 109€
- **Pérdida máxima:** -9€ (cuando ST = 100€)
- **Ganancia potencial:** Ilimitada (alta volatilidad)

### Gráfico Generado:
El código produce un gráfico que muestra:
- Perfil completo de ganancia/pérdida
- Puntos de equilibrio marcados
- Zonas de ganancia (verde) y pérdida (roja)
- Anotaciones explicativas

## 🧮 Preguntas a Responder

1. **Desembolso inicial:** ¿Cuál es la inversión total requerida?
2. **Expresiones algebraicas:** Derivar fórmulas para call y put al vencimiento
3. **Posición neta final:** Calcular el resultado total de la estrategia
4. **Condiciones de rentabilidad:** ¿Cuándo es profitable la estrategia?
5. **Implementación en Python:** Graficar el payoff (este archivo)

## 📚 Conceptos Clave

### Long Straddle
- **Estrategia:** Compra call + compra put
- **Objetivo:** Beneficiarse de alta volatilidad
- **Riesgo:** Limitado al desembolso inicial
- **Ganancia:** Potencialmente ilimitada

### Análisis de Volatilidad
La estrategia es rentable cuando:
- ST < 91€ (movimiento bajista fuerte)
- ST > 109€ (movimiento alcista fuerte)

## 💡 Tips para el Análisis

1. **Entender la forma de "V":** El payoff tiene forma de V invertida
2. **Volatilidad es clave:** La estrategia requiere movimientos significativos
3. **Tiempo es dinero:** A mayor tiempo al vencimiento, mayor prima
4. **Punto de ejercicio:** La pérdida máxima ocurre exactamente en E

## 🔍 Extensiones Posibles

- Comparar con Short Straddle
- Analizar el efecto del tiempo (Theta)
- Estudiar la volatilidad implícita
- Implementar con datos reales de mercado

---
**Fecha de entrega:** 07 de Julio, 2025  
**Formato:** PDF + código Python (.py)
