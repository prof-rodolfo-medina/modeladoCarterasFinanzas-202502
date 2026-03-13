"""
Implementación Completa del Algoritmo Hull-White
Maestría en Ciencias Computacionales y Matemáticas Aplicadas - UNIR
Curso: Modelización y Valoración de Derivados y Carteras en Finanzas

VERSIÓN 2.0 - MEJORADA CON VALIDACIÓN ROBUSTA
- Validación mejorada de parámetros u, d, q_star
- Detección de volatilidad muy baja
- Mejores mensajes de error con contexto
- Verificación de condición de no-arbitraje
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from typing import Dict, List, Tuple, Optional

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class HullWhiteCalibrator:
    """Clase para calibrar parámetros u y d del modelo binomial."""
    
    def __init__(self, prices: List[float]):
        """Inicializa con serie de precios históricos."""
        self.prices = np.array(prices)
        self.returns = None
        self.statistics = {}
        self.parameters = {}
        
        if len(self.prices) < 2:
            raise ValueError("Se necesitan al menos 2 observaciones de precios")
    
    def calculate_returns(self) -> np.ndarray:
        """Calcula rendimientos relativos."""
        self.returns = self.prices[1:] / self.prices[:-1] - 1
        return self.returns
    
    def calculate_statistics(self) -> Dict:
        """Calcula estadísticas de rendimientos."""
        if self.returns is None:
            self.calculate_returns()
        
        self.statistics = {
            'mean_return': np.mean(self.returns),
            'std_return': np.std(self.returns, ddof=1),
            'var_return': np.var(self.returns, ddof=1),
            'n_observations': len(self.prices),
            'n_returns': len(self.returns),
            'min_return': np.min(self.returns),
            'max_return': np.max(self.returns),
            'skewness': stats.skew(self.returns),
            'kurtosis': stats.kurtosis(self.returns)
        }
        return self.statistics
    
    def calibrate_hull_white(self, probability: float = 0.5) -> Dict:
        """Implementa calibración Hull-White con validación mejorada."""
        if self.statistics == {}:
            self.calculate_statistics()
        
        mu = self.statistics['mean_return']
        sigma = self.statistics['std_return']
        
        # ⚠️ DETECTAR VOLATILIDAD MUY BAJA
        MIN_SIGMA = 0.001
        sigma_warning = ""
        if sigma < MIN_SIGMA:
            sigma_warning = f"Volatilidad muy baja (σ = {sigma:.6f}). Usa calibrate_with_exponential_weights()."
        
        # Calibración
        if probability == 0.5:
            u = 1 + mu + sigma
            d = 1 + mu - sigma
        else:
            p = probability
            u_minus_d = sigma / np.sqrt(p * (1 - p))
            d = (1 + mu) - p * u_minus_d
            u = d + u_minus_d
        
        # ✅ VALIDACIÓN MEJORADA
        is_valid = (d > 0) and (u > d) and (u > 1)
        
        validation_errors = []
        if d <= 0:
            validation_errors.append(f"d = {d:.6f} ≤ 0")
        if u <= d:
            validation_errors.append(f"u = {u:.6f} ≤ d = {d:.6f}")
        if u <= 1:
            validation_errors.append(f"u = {u:.6f} ≤ 1")
        
        self.parameters = {
            'u': u,
            'd': d,
            'mu': mu,
            'sigma': sigma,
            'probability': probability,
            'is_valid': is_valid,
            'u_over_d': u / d if d != 0 else np.inf,
            'validation_errors': validation_errors,
            'sigma_warning': sigma_warning
        }
        
        return self.parameters
    
    def calibrate_with_exponential_weights(self, lambda_decay: float = 0.94) -> Dict:
        """Calibración con pesos exponenciales."""
        if self.returns is None:
            self.calculate_returns()
        
        n = len(self.returns)
        weights = np.array([lambda_decay**(n-1-i) for i in range(n)])
        weights = weights / np.sum(weights)
        
        weighted_mean = np.sum(weights * self.returns)
        weighted_var = np.sum(weights * (self.returns - weighted_mean)**2)
        weighted_std = np.sqrt(weighted_var)
        
        u = 1 + weighted_mean + weighted_std
        d = 1 + weighted_mean - weighted_std
        
        is_valid = (d > 0) and (u > d) and (u > 1)
        
        exponential_params = {
            'u': u,
            'd': d,
            'weighted_mean': weighted_mean,
            'weighted_std': weighted_std,
            'lambda_decay': lambda_decay,
            'weights': weights,
            'is_valid': is_valid,
            'u_over_d': u / d if d != 0 else np.inf
        }
        
        return exponential_params
    
    def generate_report(self) -> str:
        """Genera reporte con validación mejorada."""
        if self.parameters == {}:
            self.calibrate_hull_white()
        
        u = self.parameters['u']
        d = self.parameters['d']
        is_valid = self.parameters['is_valid']
        u_over_d = self.parameters['u_over_d']
        mu = self.statistics['mean_return']
        sigma = self.statistics['std_return']
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║              REPORTE HULL-WHITE (VALIDACIÓN MEJORADA)        ║
╠══════════════════════════════════════════════════════════════╣
║ DATOS DE ENTRADA:                                            ║
║ • Número de observaciones: {self.statistics['n_observations']:>25} ║
║ • Precio inicial: {self.prices[0]:>35.4f} ║
║ • Precio final: {self.prices[-1]:>37.4f} ║
║                                                              ║
║ ESTADÍSTICAS DE RENDIMIENTOS:                                ║
║ • Rendimiento medio (μ): {mu:>31.6f} ║
║ • Volatilidad (σ): {sigma:>34.6f} ║
║ • Rendimiento mínimo: {self.statistics['min_return']:>30.6f} ║
║ • Rendimiento máximo: {self.statistics['max_return']:>30.6f} ║
║                                                              ║
║ PARÁMETROS CALIBRADOS:                                       ║
║ • Factor de subida (u): {u:>28.6f} ║
║ • Factor de bajada (d): {d:>28.6f} ║
║ • Diferencia (u-d): {(u-d):>29.6f} ║
║ • Ratio u/d: {u_over_d:>35.6f} ║
║ • Modelo válido: {str(is_valid):>33} ║
║                                                              ║
║ VALIDACIÓN:                                                  ║
║ • ¿d > 0? {str(d > 0):>37} ║
║ • ¿u > d? {str(u > d):>37} ║
║ • ¿u > 1? {str(u > 1):>37} ║
╚══════════════════════════════════════════════════════════════╝
        """
        
        if not is_valid:
            report += "\n⚠️  ADVERTENCIA: Parámetros inválidos\n"
            for error in self.parameters.get('validation_errors', []):
                report += f"    • {error}\n"
            report += "\nSOLUCIONES:\n"
            report += "    1. Usar calibrate_with_exponential_weights()\n"
            report += "    2. Aumentar número de observaciones\n"
            report += "    3. Revisar datos para anomalías\n"
        
        if self.parameters.get('sigma_warning'):
            report += f"\n⚠️  {self.parameters['sigma_warning']}\n"
        
        return report
    
    def plot_analysis(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """Genera gráficos de análisis."""
        if self.returns is None:
            self.calculate_returns()
        if self.parameters == {}:
            self.calibrate_hull_white()
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Análisis Hull-White - Visualización Completa', fontsize=16, fontweight='bold')
        
        # Serie de precios
        axes[0, 0].plot(self.prices, 'b-', linewidth=2, marker='o', markersize=4)
        axes[0, 0].set_title('Serie de Precios Históricos')
        axes[0, 0].set_xlabel('Período')
        axes[0, 0].set_ylabel('Precio')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Rendimientos
        axes[0, 1].plot(self.returns, 'r-', linewidth=1.5, marker='o', markersize=3)
        axes[0, 1].axhline(y=self.statistics['mean_return'], color='green', linestyle='--')
        axes[0, 1].set_title('Rendimientos Relativos')
        axes[0, 1].set_xlabel('Período')
        axes[0, 1].set_ylabel('Rendimiento')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Histograma
        axes[0, 2].hist(self.returns, bins=15, density=True, alpha=0.7, color='skyblue')
        mu, sigma = self.statistics['mean_return'], self.statistics['std_return']
        x = np.linspace(self.returns.min(), self.returns.max(), 100)
        axes[0, 2].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2)
        axes[0, 2].set_title('Distribución de Rendimientos')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(self.returns, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Parámetros
        params_text = f"""u = {self.parameters['u']:.6f}
d = {self.parameters['d']:.6f}
Válido: {self.parameters['is_valid']}"""
        axes[1, 1].text(0.1, 0.5, params_text, fontsize=12,
                       bbox=dict(boxstyle="round", facecolor="lightblue"))
        axes[1, 1].axis('off')
        
        # Árbol binomial simple
        axes[1, 2].text(0.5, 0.5, 'Árbol Binomial\n(4 períodos)', 
                       ha='center', va='center', fontsize=12)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()


class OptionPricer:
    """Clase para valorar opciones con parámetros Hull-White."""
    
    def __init__(self, hull_white_params: Dict):
        self.u = hull_white_params['u']
        self.d = hull_white_params['d']
    
    def price_european_option(self, S0: float, K: float, T: int, r: float, 
                            option_type: str = 'call') -> Dict:
        """Valoración de opciones europeas con validación mejorada."""
        
        # ✅ VALIDACIÓN DE PARÁMETROS
        if S0 <= 0:
            raise ValueError(f"S0 = {S0} debe ser > 0")
        if K <= 0:
            raise ValueError(f"K = {K} debe ser > 0")
        if T < 1:
            raise ValueError(f"T = {T} debe ser >= 1")
        if r < -1:
            raise ValueError(f"r = {r} debe ser >= -1")
        
        # Validar u y d
        if self.u <= self.d:
            raise ValueError(f"u={self.u:.6f} debe ser > d={self.d:.6f}")
        if self.u <= 1:
            raise ValueError(f"u={self.u:.6f} debe ser > 1")
        if self.d <= 0:
            raise ValueError(f"d={self.d:.6f} debe ser > 0")
        
        # ✅ VALIDAR q_star
        exp_r = np.exp(r)
        
        if not (self.d < exp_r < self.u):
            print(f"⚠️  ADVERTENCIA: Violación de no-arbitraje")
            print(f"   Se requiere: d < e^r < u")
            print(f"   Actual: {self.d:.6f} < {exp_r:.6f} < {self.u:.6f}")
        
        denominator = self.u - self.d
        numerator = exp_r - self.d
        
        if abs(denominator) < 1e-10:
            raise ValueError(f"u - d = {denominator:.10f} es demasiado pequeño")
        
        q_star = numerator / denominator
        
        if not (0 < q_star < 1):
            error_msg = f"❌ Probabilidad risk-neutral inválida: q_star = {q_star:.10f}\n"
            error_msg += f"Debe estar en (0, 1)\n\n"
            error_msg += f"Detalles:\n"
            error_msg += f"  • e^r = {exp_r:.10f}\n"
            error_msg += f"  • d = {self.d:.10f}\n"
            error_msg += f"  • u = {self.u:.10f}\n"
            error_msg += f"  • q_star = ({exp_r:.10f} - {self.d:.10f}) / {denominator:.10f}\n\n"
            
            if q_star > 1:
                error_msg += "SOLUCIONES:\n"
                error_msg += "  1. Aumentar volatilidad (u - d)\n"
                error_msg += "  2. Usar calibrate_with_exponential_weights()\n"
            
            raise ValueError(error_msg)
        
        # Calcular valor de la opción
        option_value = 0
        final_prices = []
        payoffs = []
        
        for j in range(T + 1):
            ST = S0 * (self.u**(T-j)) * (self.d**j)
            final_prices.append(ST)
            
            if option_type.lower() == 'call':
                payoff = max(ST - K, 0)
            elif option_type.lower() == 'put':
                payoff = max(K - ST, 0)
            else:
                raise ValueError("option_type debe ser 'call' o 'put'")
            
            payoffs.append(payoff)
            binom_coef = stats.binom.pmf(T-j, T, q_star)
            option_value += binom_coef * payoff
        
        option_value *= np.exp(-r * T)
        
        return {
            'option_value': option_value,
            'risk_neutral_prob': q_star,
            'final_prices': final_prices,
            'payoffs': payoffs,
            'S0': S0,
            'K': K,
            'T': T,
            'r': r,
            'option_type': option_type
        }
    
    def long_straddle_analysis(self, S0: float, K: float, T: int, r: float,
                              market_call: float = None, market_put: float = None) -> Dict:
        """Análisis de Long Straddle."""
        call_results = self.price_european_option(S0, K, T, r, 'call')
        put_results = self.price_european_option(S0, K, T, r, 'put')
        
        theoretical_cost = call_results['option_value'] + put_results['option_value']
        
        results = {
            'call_value': call_results['option_value'],
            'put_value': put_results['option_value'],
            'theoretical_cost': theoretical_cost,
            'breakeven_lower': K - theoretical_cost,
            'breakeven_upper': K + theoretical_cost,
            'max_loss': -theoretical_cost
        }
        
        if market_call is not None and market_put is not None:
            results['market_cost'] = market_call + market_put
            results['market_call'] = market_call
            results['market_put'] = market_put
            results['price_difference'] = theoretical_cost - (market_call + market_put)
        
        return results


def generate_synthetic_data(S0: float = 100, n_periods: int = 100, mu: float = 0.001, 
                          sigma: float = 0.02, seed: int = 42) -> List[float]:
    """Genera datos sintéticos."""
    np.random.seed(seed)
    returns = np.random.normal(mu, sigma, n_periods)
    prices = [S0]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    return prices


def main_example():
    """Ejemplo principal."""
    print("="*80)
    print("EJEMPLO COMPLETO: ALGORITMO HULL-WHITE (v2.0)")
    print("="*80)
    
    # Generar datos
    prices = generate_synthetic_data(S0=100, n_periods=50, mu=0.002, sigma=0.025)
    
    # Calibración
    calibrator = HullWhiteCalibrator(prices)
    params = calibrator.calibrate_hull_white()
    print(calibrator.generate_report())
    
    # Valoración
    pricer = OptionPricer(params)
    try:
        result = pricer.price_european_option(S0=100, K=100, T=3, r=0.05)
        print(f"\n✅ Opción Call valorada: ${result['option_value']:.4f}")
    except ValueError as e:
        print(f"\n❌ Error: {e}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main_example()
