import numpy as np
import matplotlib.pyplot as plt

# Parámetros de la estrategia Long Straddle
C = 5  # Prima de la opción call (€)
P = 4  # Prima de la opción put (€)
E = 100  # Precio de ejercicio (€)

# Desembolso inicial total
desembolso_inicial = C + P
print(f"Desembolso inicial total: {desembolso_inicial}€")

# Rango de precios del activo subyacente al vencimiento (ST)
ST = np.linspace(70, 130, 1000)

# Función para calcular el valor de la call al vencimiento
def call_payoff(ST, E):
    return np.maximum(ST - E, 0)

# Función para calcular el valor de la put al vencimiento
def put_payoff(ST, E):
    return np.maximum(E - ST, 0)

# Calcular los payoffs individuales
call_value = call_payoff(ST, E)
put_value = put_payoff(ST, E)

# Calcular el payoff total de la estrategia (ganancia/pérdida neta)
# Payoff total = Valor call + Valor put - Desembolso inicial
payoff_total = call_value + put_value - desembolso_inicial

# Calcular puntos relevantes
# Puntos de equilibrio (breakeven points)
breakeven_lower = E - desembolso_inicial  # Punto de equilibrio inferior
breakeven_upper = E + desembolso_inicial  # Punto de equilibrio superior

# Pérdida máxima (en el precio de ejercicio)
perdida_maxima = -desembolso_inicial

print(f"\nPuntos relevantes:")
print(f"Punto de equilibrio inferior: {breakeven_lower}€")
print(f"Punto de equilibrio superior: {breakeven_upper}€")
print(f"Pérdida máxima: {perdida_maxima}€ (cuando ST = {E}€)")

# Crear el gráfico
plt.figure(figsize=(12, 8))

# Graficar el payoff total
plt.plot(ST, payoff_total, 'b-', linewidth=2.5, label='Long Straddle Payoff')

# Graficar línea de equilibrio (ganancia/pérdida = 0)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.7, label='Línea de equilibrio')

# Marcar puntos relevantes
plt.plot(breakeven_lower, 0, 'ro', markersize=8, label=f'Breakeven inferior ({breakeven_lower}€)')
plt.plot(breakeven_upper, 0, 'ro', markersize=8, label=f'Breakeven superior ({breakeven_upper}€)')
plt.plot(E, perdida_maxima, 'rs', markersize=8, label=f'Pérdida máxima ({E}€, {perdida_maxima}€)')

# Sombrear áreas de ganancia y pérdida
plt.fill_between(ST, payoff_total, 0, where=(payoff_total > 0), 
                 alpha=0.3, color='green', label='Zona de ganancia')
plt.fill_between(ST, payoff_total, 0, where=(payoff_total < 0), 
                 alpha=0.3, color='red', label='Zona de pérdida')

# Configurar el gráfico
plt.xlabel('Precio del activo subyacente al vencimiento (ST) €', fontsize=12)
plt.ylabel('Ganancia/Pérdida (€)', fontsize=12)
plt.title('Perfil de Ganancia/Pérdida - Estrategia Long Straddle\n' + 
          f'Call Premium: {C}€, Put Premium: {P}€, Strike Price: {E}€', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

# Establecer límites del gráfico
plt.xlim(70, 130)
plt.ylim(-12, 25)

# Agregar anotaciones para mayor claridad
plt.annotate(f'Desembolso inicial: {desembolso_inicial}€', 
             xy=(75, 20), fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat"))

plt.annotate('La estrategia es rentable cuando\nhay alta volatilidad del precio', 
             xy=(85, 15), fontsize=9, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

plt.tight_layout()
plt.show()

# Análisis adicional
print(f"\nAnálisis de la estrategia:")
print(f"• La estrategia Long Straddle es profitable cuando ST < {breakeven_lower}€ o ST > {breakeven_upper}€")
print(f"• La pérdida máxima de {perdida_maxima}€ ocurre cuando ST = {E}€ (precio de ejercicio)")
print(f"• No hay límite superior para las ganancias potenciales")
print(f"• La estrategia se beneficia de alta volatilidad del activo subyacente")

# Calcular algunos ejemplos específicos
ejemplos_ST = [85, 95, 100, 105, 115]
print(f"\nEjemplos de payoff para diferentes precios al vencimiento:")
for st in ejemplos_ST:
    call_val = max(st - E, 0)
    put_val = max(E - st, 0)
    payoff = call_val + put_val - desembolso_inicial
    print(f"ST = {st}€: Call = {call_val}€, Put = {put_val}€, Payoff total = {payoff}€")
