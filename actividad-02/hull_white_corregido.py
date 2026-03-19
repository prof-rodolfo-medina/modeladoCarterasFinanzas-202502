"""
Implementación del Modelo Hull-White con Datos Validados
Universidad Internacional de La Rioja (UNIR)
Dr. Rodolfo Rafael Medina Ramírez

CORRECCIONES:
- Parámetros realistas calibrados
- Curva de tipos suavizada
- Validación de inputs
- Manejo de errores mejorado
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class HullWhiteModel:
    """
    Modelo Hull-White de 1 factor para tasas de interés.
    
    dr(t) = [θ(t) - α·r(t)]dt + σ·dW(t)
    
    Parámetros:
    -----------
    alpha : float
        Velocidad de reversión a la media (típicamente 0.01 - 0.5)
    sigma : float
        Volatilidad de la tasa corta (típicamente 0.005 - 0.02)
    """
    
    def __init__(self, alpha=0.1, sigma=0.01):
        """
        Inicializa el modelo Hull-White.
        
        Parameters:
        -----------
        alpha : float
            Velocidad de reversión (recomendado: 0.05 - 0.15)
        sigma : float
            Volatilidad (recomendado: 0.005 - 0.015)
        """
        # Validar parámetros
        if alpha <= 0 or alpha > 1:
            raise ValueError(f"alpha debe estar en (0, 1]. Valor: {alpha}")
        if sigma <= 0 or sigma > 0.1:
            raise ValueError(f"sigma debe estar en (0, 0.1]. Valor: {sigma}")
        
        self.alpha = alpha
        self.sigma = sigma
        print(f"✅ Modelo Hull-White inicializado:")
        print(f"   α (reversión) = {alpha}")
        print(f"   σ (volatilidad) = {sigma}")
    
    def theta(self, t, forward_curve):
        """
        Calcula θ(t) para ajustarse a la curva forward.
        
        θ(t) = ∂f(0,t)/∂t + α·f(0,t) + (σ²/2α)·(1 - e^(-2αt))
        """
        # Derivada numérica de la curva forward
        dt = 0.001
        if t < dt:
            df_dt = (forward_curve(t + dt) - forward_curve(t)) / dt
        else:
            df_dt = (forward_curve(t) - forward_curve(t - dt)) / dt
        
        f_t = forward_curve(t)
        
        # Término de ajuste de volatilidad
        vol_adjustment = (self.sigma**2 / (2 * self.alpha)) * (1 - np.exp(-2 * self.alpha * t))
        
        return df_dt + self.alpha * f_t + vol_adjustment
    
    def simulate_paths(self, r0, T, n_steps, n_paths, forward_curve):
        """
        Simula trayectorias de la tasa corta usando Euler-Maruyama.
        
        Parameters:
        -----------
        r0 : float
            Tasa inicial
        T : float
            Horizonte temporal (años)
        n_steps : int
            Número de pasos de tiempo
        n_paths : int
            Número de trayectorias a simular
        forward_curve : callable
            Función que retorna f(0,t)
            
        Returns:
        --------
        times : np.ndarray
            Vector de tiempos
        rates : np.ndarray
            Matriz (n_steps, n_paths) de tasas simuladas
        """
        dt = T / n_steps
        times = np.linspace(0, T, n_steps)
        rates = np.zeros((n_steps, n_paths))
        rates[0, :] = r0
        
        # Pre-calcular theta para eficiencia
        theta_values = np.array([self.theta(t, forward_curve) for t in times])
        
        # Simular trayectorias
        np.random.seed(42)
        dW = np.random.normal(0, np.sqrt(dt), (n_steps-1, n_paths))
        
        for i in range(1, n_steps):
            t = times[i-1]
            r = rates[i-1, :]
            
            # Esquema Euler-Maruyama
            drift = (theta_values[i-1] - self.alpha * r) * dt
            diffusion = self.sigma * dW[i-1, :]
            
            rates[i, :] = r + drift + diffusion
            
            # Asegurar tasas no negativas (floor en 0)
            rates[i, :] = np.maximum(rates[i, :], 0.0001)
        
        return times, rates
    
    def bond_price(self, r, t, T):
        """
        Calcula el precio de un bono cupón cero.
        
        P(t,T) = A(t,T) · exp[-B(t,T)·r(t)]
        
        Parameters:
        -----------
        r : float or array
            Tasa corta actual
        t : float
            Tiempo actual
        T : float
            Madurez del bono
            
        Returns:
        --------
        float or array : Precio del bono
        """
        tau = T - t
        
        # B(t,T)
        B = (1 / self.alpha) * (1 - np.exp(-self.alpha * tau))
        
        # A(t,T) - versión simplificada
        # En producción, esto debería ajustarse a la curva inicial
        variance_term = (self.sigma**2 / (2 * self.alpha**2)) * \
                       (tau - B - (self.sigma**2 * B**2) / (4 * self.alpha))
        
        A = np.exp(variance_term)
        
        return A * np.exp(-B * r)


def generar_curva_forward_realista():
    """
    Genera una curva forward realista basada en Nelson-Siegel.
    
    f(t) = β₀ + β₁·exp(-t/τ) + β₂·(t/τ)·exp(-t/τ)
    """
    # Parámetros Nelson-Siegel calibrados (curva típica)
    beta0 = 0.03   # Nivel largo plazo
    beta1 = -0.01  # Pendiente
    beta2 = 0.02   # Curvatura
    tau = 2.0      # Factor de decaimiento
    
    def forward_curve(t):
        return beta0 + beta1 * np.exp(-t/tau) + beta2 * (t/tau) * np.exp(-t/tau)
    
    return forward_curve


def ejemplo_completo():
    """Ejemplo completo con visualización."""
    
    print("=" * 70)
    print("MODELO HULL-WHITE - SIMULACIÓN COMPLETA")
    print("=" * 70)
    
    # 1. Configurar parámetros
    print("\n📊 CONFIGURACIÓN:")
    alpha = 0.1      # Reversión a la media
    sigma = 0.01     # Volatilidad
    r0 = 0.02        # Tasa inicial (2%)
    T = 10.0         # Horizonte 10 años
    n_steps = 1000   # Pasos de tiempo
    n_paths = 100    # Trayectorias
    
    print(f"   Tasa inicial: {r0*100:.2f}%")
    print(f"   Horizonte: {T} años")
    print(f"   Trayectorias: {n_paths}")
    print(f"   Pasos: {n_steps}")
    
    # 2. Crear modelo
    print("\n🔧 Inicializando modelo...")
    hw = HullWhiteModel(alpha=alpha, sigma=sigma)
    
    # 3. Generar curva forward
    print("\n📈 Generando curva forward...")
    forward_curve = generar_curva_forward_realista()
    
    # Verificar curva
    t_test = np.linspace(0, T, 50)
    f_test = [forward_curve(t) for t in t_test]
    print(f"   Rango de tasas forward: {min(f_test)*100:.2f}% - {max(f_test)*100:.2f}%")
    
    # 4. Simular trayectorias
    print("\n🎲 Simulando trayectorias...")
    times, rates = hw.simulate_paths(r0, T, n_steps, n_paths, forward_curve)
    print(f"   ✅ Simulación completada")
    print(f"   Tasa final promedio: {rates[-1, :].mean()*100:.2f}%")
    print(f"   Tasa final min/max: {rates[-1, :].min()*100:.2f}% / {rates[-1, :].max()*100:.2f}%")
    
    # 5. Visualización
    print("\n📊 Generando gráficas...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Gráfica 1: Trayectorias de tasas
    ax1 = axes[0, 0]
    # Mostrar solo algunas trayectorias para claridad
    for i in range(min(20, n_paths)):
        ax1.plot(times, rates[:, i] * 100, alpha=0.3, linewidth=0.5)
    ax1.plot(times, rates.mean(axis=1) * 100, 'r-', linewidth=2, label='Media')
    ax1.plot(t_test, np.array(f_test) * 100, 'k--', linewidth=2, label='Forward inicial')
    ax1.set_xlabel('Tiempo (años)', fontweight='bold')
    ax1.set_ylabel('Tasa (%)', fontweight='bold')
    ax1.set_title('Trayectorias de la Tasa Corta', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfica 2: Distribución final
    ax2 = axes[0, 1]
    ax2.hist(rates[-1, :] * 100, bins=30, alpha=0.7, edgecolor='black')
    ax2.axvline(rates[-1, :].mean() * 100, color='red', linestyle='--', 
               linewidth=2, label=f'Media: {rates[-1, :].mean()*100:.2f}%')
    ax2.set_xlabel('Tasa (%)', fontweight='bold')
    ax2.set_ylabel('Frecuencia', fontweight='bold')
    ax2.set_title(f'Distribución Final (t={T} años)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Gráfica 3: Curva forward inicial
    ax3 = axes[1, 0]
    ax3.plot(t_test, np.array(f_test) * 100, 'b-', linewidth=2)
    ax3.set_xlabel('Madurez (años)', fontweight='bold')
    ax3.set_ylabel('Tasa Forward (%)', fontweight='bold')
    ax3.set_title('Curva Forward Inicial', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Gráfica 4: Evolución de media y desviación estándar
    ax4 = axes[1, 1]
    mean_rates = rates.mean(axis=1) * 100
    std_rates = rates.std(axis=1) * 100
    ax4.plot(times, mean_rates, 'b-', linewidth=2, label='Media')
    ax4.fill_between(times, mean_rates - std_rates, mean_rates + std_rates,
                     alpha=0.3, label='±1 σ')
    ax4.set_xlabel('Tiempo (años)', fontweight='bold')
    ax4.set_ylabel('Tasa (%)', fontweight='bold')
    ax4.set_title('Evolución de Media y Desviación Estándar', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Modelo Hull-White: Análisis Completo', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Guardar
    output_file = 'hull_white_simulacion.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✅ Gráfica guardada: {output_file}")
    
    plt.show()
    
    print("\n" + "=" * 70)
    print("✅ SIMULACIÓN COMPLETADA EXITOSAMENTE")
    print("=" * 70)


if __name__ == "__main__":
    try:
        ejemplo_completo()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


