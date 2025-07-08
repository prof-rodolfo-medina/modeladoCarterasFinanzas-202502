"""
Implementación Completa del Algoritmo Hull-White
Maestría en Ciencias Computacionales y Matemáticas Aplicadas - UNIR
Curso: Modelización y Valoración de Derivados y Carteras en Finanzas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from typing import Dict, List, Tuple, Optional

# Configuración de estilo para gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class HullWhiteCalibrator:
    """
    Clase para implementar el algoritmo Hull-White de calibración de parámetros
    binomiales u y d a partir de datos históricos.
    """
    
    def __init__(self, prices: List[float]):
        """
        Inicializa el calibrador con una serie de precios.
        
        Args:
            prices: Lista de precios históricos del activo subyacente
        """
        self.prices = np.array(prices)
        self.returns = None
        self.statistics = {}
        self.parameters = {}
        
        if len(self.prices) < 2:
            raise ValueError("Se necesitan al menos 2 observaciones de precios")
    
    def calculate_returns(self) -> np.ndarray:
        """Calcula los rendimientos relativos de la serie de precios."""
        self.returns = self.prices[1:] / self.prices[:-1] - 1
        return self.returns
    
    def calculate_statistics(self) -> Dict:
        """Calcula estadísticas descriptivas de los rendimientos."""
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
        """
        Implementa el algoritmo Hull-White para calibrar u y d.
        
        Args:
            probability: Probabilidad de movimiento hacia arriba (default 0.5)
            
        Returns:
            Diccionario con parámetros calibrados
        """
        if self.statistics == {}:
            self.calculate_statistics()
        
        mu = self.statistics['mean_return']
        sigma = self.statistics['std_return']
        
        # Calibración Hull-White estándar (p = 0.5)
        if probability == 0.5:
            u = 1 + mu + sigma
            d = 1 + mu - sigma
        else:
            # Versión generalizada con probabilidad diferente
            # Resolver el sistema de ecuaciones
            # 1 + μΔt = pu + (1-p)d
            # σ²Δt = p(1-p)(u-d)²
            
            p = probability
            # De la segunda ecuación: u - d = σ√(Δt/[p(1-p)])
            u_minus_d = sigma / np.sqrt(p * (1 - p))
            
            # De la primera ecuación: u + d = 2(1 + μΔt) - (u-d)p - (u-d)(1-p)
            # Simplificando: pu + (1-p)d = 1 + μΔt
            # Como u = d + (u-d), entonces: p(d + u-d) + (1-p)d = 1 + μΔt
            # pd + p(u-d) + (1-p)d = 1 + μΔt
            # d + p(u-d) = 1 + μΔt
            
            d = (1 + mu) - p * u_minus_d
            u = d + u_minus_d
        
        # Validación de parámetros
        is_valid = (d > 0) and (u > d) and (u > 1)
        
        self.parameters = {
            'u': u,
            'd': d,
            'mu': mu,
            'sigma': sigma,
            'probability': probability,
            'is_valid': is_valid,
            'u_over_d': u / d if d != 0 else np.inf
        }
        
        return self.parameters
    
    def calibrate_with_exponential_weights(self, lambda_decay: float = 0.94) -> Dict:
        """
        Calibración Hull-White con pesos exponenciales para dar más importancia
        a observaciones más recientes.
        
        Args:
            lambda_decay: Factor de decaimiento exponencial (0 < λ < 1)
            
        Returns:
            Diccionario con parámetros calibrados
        """
        if self.returns is None:
            self.calculate_returns()
        
        n = len(self.returns)
        
        # Calcular pesos exponenciales (más peso a datos recientes)
        weights = np.array([lambda_decay**(n-1-i) for i in range(n)])
        weights = weights / np.sum(weights)  # Normalizar
        
        # Estadísticas ponderadas
        weighted_mean = np.sum(weights * self.returns)
        weighted_var = np.sum(weights * (self.returns - weighted_mean)**2)
        weighted_std = np.sqrt(weighted_var)
        
        # Parámetros Hull-White
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
        """Genera un reporte completo del análisis."""
        if self.parameters == {}:
            self.calibrate_hull_white()
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                   REPORTE HULL-WHITE                         ║
╠══════════════════════════════════════════════════════════════╣
║ DATOS DE ENTRADA:                                            ║
║ • Número de observaciones: {self.statistics['n_observations']:>25} ║
║ • Precio inicial: {self.prices[0]:>35.4f} ║
║ • Precio final: {self.prices[-1]:>37.4f} ║
║                                                              ║
║ ESTADÍSTICAS DE RENDIMIENTOS:                                ║
║ • Rendimiento medio (μΔt): {self.statistics['mean_return']:>25.6f} ║
║ • Volatilidad (σ√Δt): {self.statistics['std_return']:>30.6f} ║
║ • Rendimiento mínimo: {self.statistics['min_return']:>30.6f} ║
║ • Rendimiento máximo: {self.statistics['max_return']:>30.6f} ║
║ • Asimetría: {self.statistics['skewness']:>39.6f} ║
║ • Curtosis: {self.statistics['kurtosis']:>40.6f} ║
║                                                              ║
║ PARÁMETROS CALIBRADOS:                                       ║
║ • Factor de subida (u): {self.parameters['u']:>28.6f} ║
║ • Factor de bajada (d): {self.parameters['d']:>28.6f} ║
║ • Ratio u/d: {self.parameters['u_over_d']:>35.6f} ║
║ • Modelo válido: {str(self.parameters['is_valid']):>33} ║
╚══════════════════════════════════════════════════════════════╝
        """
        
        return report
    
    def plot_analysis(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """Genera gráficos de análisis completo."""
        if self.returns is None:
            self.calculate_returns()
        if self.parameters == {}:
            self.calibrate_hull_white()
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Análisis Hull-White - Visualización Completa', fontsize=16, fontweight='bold')
        
        # 1. Serie de precios
        axes[0, 0].plot(self.prices, 'b-', linewidth=2, marker='o', markersize=4)
        axes[0, 0].set_title('Serie de Precios Históricos')
        axes[0, 0].set_xlabel('Período')
        axes[0, 0].set_ylabel('Precio')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Serie de rendimientos
        axes[0, 1].plot(self.returns, 'r-', linewidth=1.5, marker='o', markersize=3)
        axes[0, 1].axhline(y=self.statistics['mean_return'], color='green', 
                          linestyle='--', label=f'Media: {self.statistics["mean_return"]:.4f}')
        axes[0, 1].set_title('Rendimientos Relativos')
        axes[0, 1].set_xlabel('Período')
        axes[0, 1].set_ylabel('Rendimiento')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Histograma de rendimientos
        axes[0, 2].hist(self.returns, bins=15, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Ajustar distribución normal
        mu, sigma = self.statistics['mean_return'], self.statistics['std_return']
        x = np.linspace(self.returns.min(), self.returns.max(), 100)
        axes[0, 2].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal ajustada')
        axes[0, 2].axvline(mu, color='green', linestyle='--', label=f'Media: {mu:.4f}')
        axes[0, 2].set_title('Distribución de Rendimientos')
        axes[0, 2].set_xlabel('Rendimiento')
        axes[0, 2].set_ylabel('Densidad')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Q-Q plot para normalidad
        stats.probplot(self.returns, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Test de Normalidad)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Parámetros calibrados
        params_text = f"""Parámetros Hull-White:
        
u = {self.parameters['u']:.6f}
d = {self.parameters['d']:.6f}
μ = {self.parameters['mu']:.6f}
σ = {self.parameters['sigma']:.6f}

Ratio u/d = {self.parameters['u_over_d']:.4f}
Válido: {self.parameters['is_valid']}"""
        
        axes[1, 1].text(0.1, 0.5, params_text, fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title('Parámetros Calibrados')
        axes[1, 1].axis('off')
        
        # 6. Simulación de árbol binomial
        self._plot_binomial_tree(axes[1, 2])
        
        plt.tight_layout()
        plt.show()
    
    def _plot_binomial_tree(self, ax, periods: int = 4):
        """Plotea un árbol binomial de ejemplo con los parámetros calibrados."""
        S0 = self.prices[0]
        u, d = self.parameters['u'], self.parameters['d']
        
        # Generar árbol
        tree_values = {}
        for i in range(periods + 1):
            tree_values[i] = []
            for j in range(i + 1):
                value = S0 * (u**(i-j)) * (d**j)
                tree_values[i].append(value)
        
        # Plotear
        for period, values in tree_values.items():
            y_positions = np.linspace(-period/2, period/2, len(values))
            ax.scatter([period] * len(values), y_positions, s=100, c='red', alpha=0.7)
            
            for j, (y_pos, value) in enumerate(zip(y_positions, values)):
                ax.annotate(f'{value:.2f}', (period, y_pos), xytext=(5, 0), 
                           textcoords='offset points', fontsize=8, ha='left')
        
        ax.set_xlabel('Período')
        ax.set_ylabel('Posición')
        ax.set_title(f'Árbol Binomial ({periods} períodos)')
        ax.grid(True, alpha=0.3)


class OptionPricer:
    """Clase para valorar opciones usando parámetros Hull-White."""
    
    def __init__(self, hull_white_params: Dict):
        """
        Inicializa el valorador con parámetros Hull-White.
        
        Args:
            hull_white_params: Diccionario con parámetros u, d calibrados
        """
        self.u = hull_white_params['u']
        self.d = hull_white_params['d']
    
    def price_european_option(self, S0: float, K: float, T: int, r: float, 
                            option_type: str = 'call') -> Dict:
        """
        Valoración de opciones europeas con el método binomial.
        
        Args:
            S0: Precio inicial del subyacente
            K: Precio de ejercicio
            T: Número de períodos hasta vencimiento
            r: Tasa libre de riesgo por período
            option_type: 'call' o 'put'
            
        Returns:
            Diccionario con resultados de valoración
        """
        # Probabilidad risk-neutral
        q_star = (np.exp(r) - self.d) / (self.u - self.d)
        
        if q_star <= 0 or q_star >= 1:
            raise ValueError(f"Probabilidad risk-neutral inválida: {q_star}")
        
        # Precios finales y payoffs
        final_prices = []
        payoffs = []
        contributions = []
        
        option_value = 0
        
        for j in range(T + 1):
            # Precio final
            ST = S0 * (self.u**(T-j)) * (self.d**j)
            final_prices.append(ST)
            
            # Payoff
            if option_type.lower() == 'call':
                payoff = max(ST - K, 0)
            elif option_type.lower() == 'put':
                payoff = max(K - ST, 0)
            else:
                raise ValueError("option_type debe ser 'call' o 'put'")
            
            payoffs.append(payoff)
            
            # Contribución al precio
            binom_coef = stats.binom.pmf(T-j, T, q_star)
            contribution = binom_coef * payoff
            contributions.append(contribution)
            
            option_value += contribution
        
        # Actualizar a valor presente
        option_value *= np.exp(-r * T)
        
        return {
            'option_value': option_value,
            'risk_neutral_prob': q_star,
            'final_prices': final_prices,
            'payoffs': payoffs,
            'contributions': contributions,
            'S0': S0,
            'K': K,
            'T': T,
            'r': r,
            'option_type': option_type
        }
    
    def long_straddle_analysis(self, S0: float, K: float, T: int, r: float,
                              market_call: float = None, market_put: float = None) -> Dict:
        """
        Análisis completo de estrategia Long Straddle.
        
        Args:
            S0: Precio inicial del subyacente
            K: Precio de ejercicio
            T: Períodos hasta vencimiento
            r: Tasa libre de riesgo
            market_call: Prima de call en el mercado (opcional)
            market_put: Prima de put en el mercado (opcional)
            
        Returns:
            Diccionario con análisis completo
        """
        # Valorar call y put
        call_results = self.price_european_option(S0, K, T, r, 'call')
        put_results = self.price_european_option(S0, K, T, r, 'put')
        
        # Costos de la estrategia
        theoretical_cost = call_results['option_value'] + put_results['option_value']
        
        # Puntos de equilibrio
        breakeven_lower = K - theoretical_cost
        breakeven_upper = K + theoretical_cost
        
        # Análisis de payoff
        ST_range = np.linspace(0.5 * K, 1.5 * K, 100)
        straddle_payoffs = []
        
        for ST in ST_range:
            call_payoff = max(ST - K, 0)
            put_payoff = max(K - ST, 0)
            net_payoff = call_payoff + put_payoff - theoretical_cost
            straddle_payoffs.append(net_payoff)
        
        results = {
            'call_value': call_results['option_value'],
            'put_value': put_results['option_value'],
            'theoretical_cost': theoretical_cost,
            'breakeven_lower': breakeven_lower,
            'breakeven_upper': breakeven_upper,
            'max_loss': -theoretical_cost,
            'ST_range': ST_range,
            'straddle_payoffs': straddle_payoffs,
            'call_results': call_results,
            'put_results': put_results
        }
        
        # Comparación con precios de mercado si están disponibles
        if market_call is not None and market_put is not None:
            market_cost = market_call + market_put
            results['market_cost'] = market_cost
            results['market_call'] = market_call
            results['market_put'] = market_put
            results['price_difference'] = theoretical_cost - market_cost
            results['call_diff'] = call_results['option_value'] - market_call
            results['put_diff'] = put_results['option_value'] - market_put
        
        return results
    
    def plot_straddle_payoff(self, straddle_results: Dict, figsize: Tuple[int, int] = (12, 8)):
        """Grafica el perfil de payoff del Long Straddle."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Gráfico principal de payoff
        ST_range = straddle_results['ST_range']
        payoffs = straddle_results['straddle_payoffs']
        K = straddle_results['call_results']['K']
        
        ax1.plot(ST_range, payoffs, 'b-', linewidth=3, label='Long Straddle')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.7, label='Línea de equilibrio')
        ax1.axvline(x=K, color='gray', linestyle=':', alpha=0.7, label=f'Strike: {K}')
        
        # Marcar puntos clave
        ax1.plot(straddle_results['breakeven_lower'], 0, 'ro', markersize=10, 
                label=f'BE Inferior: {straddle_results["breakeven_lower"]:.2f}')
        ax1.plot(straddle_results['breakeven_upper'], 0, 'ro', markersize=10,
                label=f'BE Superior: {straddle_results["breakeven_upper"]:.2f}')
        ax1.plot(K, straddle_results['max_loss'], 'rs', markersize=10,
                label=f'Pérdida máx: {straddle_results["max_loss"]:.2f}')
        
        # Áreas de ganancia y pérdida
        ax1.fill_between(ST_range, payoffs, 0, where=(np.array(payoffs) > 0), 
                        alpha=0.3, color='green', label='Zona ganancia')
        ax1.fill_between(ST_range, payoffs, 0, where=(np.array(payoffs) < 0), 
                        alpha=0.3, color='red', label='Zona pérdida')
        
        ax1.set_xlabel('Precio del Subyacente al Vencimiento (ST)')
        ax1.set_ylabel('Ganancia/Pérdida')
        ax1.set_title('Perfil de Payoff - Long Straddle')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico de comparación teórico vs mercado (si disponible)
        if 'market_cost' in straddle_results:
            categories = ['Call', 'Put', 'Total']
            theoretical = [straddle_results['call_value'], 
                          straddle_results['put_value'],
                          straddle_results['theoretical_cost']]
            market = [straddle_results['market_call'],
                     straddle_results['market_put'], 
                     straddle_results['market_cost']]
            
            x = np.arange(len(categories))
            width = 0.35
            
            ax2.bar(x - width/2, theoretical, width, label='Hull-White', alpha=0.8)
            ax2.bar(x + width/2, market, width, label='Mercado', alpha=0.8)
            
            ax2.set_xlabel('Componente')
            ax2.set_ylabel('Precio')
            ax2.set_title('Comparación: Teórico vs Mercado')
            ax2.set_xticks(x)
            ax2.set_xticklabels(categories)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Añadir valores en las barras
            for i, (theo, mark) in enumerate(zip(theoretical, market)):
                ax2.text(i - width/2, theo + 0.1, f'{theo:.3f}', ha='center', va='bottom')
                ax2.text(i + width/2, mark + 0.1, f'{mark:.3f}', ha='center', va='bottom')
        else:
            # Si no hay datos de mercado, mostrar componentes individuales
            components = ['Call Payoff', 'Put Payoff', 'Costo Total']
            values = [straddle_results['call_value'], 
                     straddle_results['put_value'],
                     -straddle_results['theoretical_cost']]
            colors = ['blue', 'orange', 'red']
            
            bars = ax2.bar(components, values, color=colors, alpha=0.7)
            ax2.set_ylabel('Valor')
            ax2.set_title('Componentes del Long Straddle')
            ax2.grid(True, alpha=0.3)
            
            # Añadir valores en las barras
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1 if height > 0 else height - 0.3,
                        f'{value:.3f}', ha='center', va='bottom' if height > 0 else 'top')
        
        plt.tight_layout()
        plt.show()


def sensitivity_analysis(prices: List[float], min_obs: int = 10, step: int = 5) -> pd.DataFrame:
    """
    Análisis de sensibilidad del algoritmo Hull-White al tamaño de muestra.
    
    Args:
        prices: Serie de precios completa
        min_obs: Mínimo número de observaciones
        step: Paso para incrementar observaciones
        
    Returns:
        DataFrame con resultados de sensibilidad
    """
    results = []
    max_obs = len(prices)
    
    for n_obs in range(min_obs, max_obs + 1, step):
        subsample = prices[-n_obs:]  # Usar las últimas n_obs observaciones
        
        try:
            calibrator = HullWhiteCalibrator(subsample)
            params = calibrator.calibrate_hull_white()
            stats = calibrator.calculate_statistics()
            
            results.append({
                'n_obs': n_obs,
                'u': params['u'],
                'd': params['d'],
                'mu': params['mu'],
                'sigma': params['sigma'],
                'ratio_ud': params['u_over_d'],
                'is_valid': params['is_valid'],
                'skewness': stats['skewness'],
                'kurtosis': stats['kurtosis']
            })
        except Exception as e:
            print(f"Error con {n_obs} observaciones: {e}")
    
    return pd.DataFrame(results)


def plot_sensitivity_analysis(sensitivity_df: pd.DataFrame, figsize: Tuple[int, int] = (15, 10)):
    """Visualiza los resultados del análisis de sensibilidad."""
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Análisis de Sensibilidad - Algoritmo Hull-White', fontsize=16)
    
    # Factor u
    axes[0, 0].plot(sensitivity_df['n_obs'], sensitivity_df['u'], 'b-o', linewidth=2, markersize=4)
    axes[0, 0].set_title('Factor u vs Tamaño de Muestra')
    axes[0, 0].set_xlabel('Número de Observaciones')
    axes[0, 0].set_ylabel('Factor u')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Factor d
    axes[0, 1].plot(sensitivity_df['n_obs'], sensitivity_df['d'], 'r-o', linewidth=2, markersize=4)
    axes[0, 1].set_title('Factor d vs Tamaño de Muestra')
    axes[0, 1].set_xlabel('Número de Observaciones')
    axes[0, 1].set_ylabel('Factor d')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Volatilidad
    axes[0, 2].plot(sensitivity_df['n_obs'], sensitivity_df['sigma'], 'g-o', linewidth=2, markersize=4)
    axes[0, 2].set_title('Volatilidad vs Tamaño de Muestra')
    axes[0, 2].set_xlabel('Número de Observaciones')
    axes[0, 2].set_ylabel('Volatilidad (σ)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Ratio u/d
    axes[1, 0].plot(sensitivity_df['n_obs'], sensitivity_df['ratio_ud'], 'm-o', linewidth=2, markersize=4)
    axes[1, 0].set_title('Ratio u/d vs Tamaño de Muestra')
    axes[1, 0].set_xlabel('Número de Observaciones')
    axes[1, 0].set_ylabel('u/d')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Rendimiento medio
    axes[1, 1].plot(sensitivity_df['n_obs'], sensitivity_df['mu'], 'c-o', linewidth=2, markersize=4)
    axes[1, 1].set_title('Rendimiento Medio vs Tamaño de Muestra')
    axes[1, 1].set_xlabel('Número de Observaciones')
    axes[1, 1].set_ylabel('μ')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Validez del modelo
    valid_counts = sensitivity_df.groupby('n_obs')['is_valid'].sum()
    axes[1, 2].bar(valid_counts.index, valid_counts.values, alpha=0.7, color='lightgreen')
    axes[1, 2].set_title('Modelos Válidos por Tamaño de Muestra')
    axes[1, 2].set_xlabel('Número de Observaciones')
    axes[1, 2].set_ylabel('Modelos Válidos')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def generate_synthetic_data(S0: float = 100, n_periods: int = 100, mu: float = 0.001, 
                          sigma: float = 0.02, seed: int = 42) -> List[float]:
    """
    Genera datos sintéticos para pruebas del algoritmo Hull-White.
    
    Args:
        S0: Precio inicial
        n_periods: Número de períodos
        mu: Deriva del proceso
        sigma: Volatilidad del proceso
        seed: Semilla para reproducibilidad
        
    Returns:
        Lista de precios simulados
    """
    np.random.seed(seed)
    returns = np.random.normal(mu, sigma, n_periods)
    
    prices = [S0]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    
    return prices


def main_example():
    """Ejemplo principal de uso de las clases y funciones."""
    
    print("="*80)
    print("EJEMPLO COMPLETO: ALGORITMO HULL-WHITE")
    print("="*80)
    
    # 1. Generar datos sintéticos
    print("\n1. Generando datos sintéticos...")
    prices = generate_synthetic_data(S0=100, n_periods=50, mu=0.002, sigma=0.025)
    
    # 2. Calibración Hull-White
    print("\n2. Calibración Hull-White...")
    calibrator = HullWhiteCalibrator(prices)
    params = calibrator.calibrate_hull_white()
    
    # 3. Mostrar reporte
    print(calibrator.generate_report())
    
    # 4. Análisis gráfico
    print("\n3. Generando análisis gráfico...")
    calibrator.plot_analysis()
    
    # 5. Valoración de opciones
    print("\n4. Valoración de opciones...")
    pricer = OptionPricer(params)
    
    # Parámetros de la opción
    S0 = 100
    K = 100
    T = 3
    r = 0.05
    
    # Análisis Long Straddle
    straddle_results = pricer.long_straddle_analysis(
        S0=S0, K=K, T=T, r=r,
        market_call=5.0, market_put=4.0  # Precios del ejercicio original
    )
    
    print(f"\nResultados Long Straddle:")
    print(f"Call teórica: {straddle_results['call_value']:.4f} vs Mercado: {straddle_results['market_call']:.4f}")
    print(f"Put teórica: {straddle_results['put_value']:.4f} vs Mercado: {straddle_results['market_put']:.4f}")
    print(f"Costo total teórico: {straddle_results['theoretical_cost']:.4f}")
    print(f"Costo total mercado: {straddle_results['market_cost']:.4f}")
    print(f"Diferencia: {straddle_results['price_difference']:.4f}")
    print(f"Breakeven inferior: {straddle_results['breakeven_lower']:.2f}")
    print(f"Breakeven superior: {straddle_results['breakeven_upper']:.2f}")
    
    # Gráfico del straddle
    pricer.plot_straddle_payoff(straddle_results)
    
    # 6. Análisis de sensibilidad
    print("\n5. Análisis de sensibilidad...")
    sensitivity_df = sensitivity_analysis(prices, min_obs=15, step=3)
    plot_sensitivity_analysis(sensitivity_df)
    
    print("\nResumen de sensibilidad:")
    print(sensitivity_df.describe())
    
    print("\n" + "="*80)
    print("ANÁLISIS COMPLETADO")
    print("="*80)


if __name__ == "__main__":
    # Ejecutar ejemplo principal
    main_example()
    
    # Ejemplo adicional con datos del ejercicio original
    print("\n" + "="*80)
    print("EJEMPLO CON DATOS DEL EJERCICIO ORIGINAL")
    print("="*80)
    
    # Datos de ejemplo del documento
    exercise_prices = [100, 102, 98, 105, 103, 99, 107, 104, 101, 108, 106, 103]
    
    calibrator_ex = HullWhiteCalibrator(exercise_prices)
    params_ex = calibrator_ex.calibrate_hull_white()
    
    print(calibrator_ex.generate_report())
    
    # Comparar con parámetros dados en el ejercicio
    # En el código original: C=5, P=4, E=100
    pricer_ex = OptionPricer(params_ex)
    straddle_ex = pricer_ex.long_straddle_analysis(
        S0=100, K=100, T=3, r=0.05,
        market_call=5.0, market_put=4.0
    )
    
    print(f"\nComparación con ejercicio original:")
    print(f"Parámetros calibrados: u={params_ex['u']:.6f}, d={params_ex['d']:.6f}")
    print(f"Call Hull-White: {straddle_ex['call_value']:.4f} vs Original: 5.00")
    print(f"Put Hull-White: {straddle_ex['put_value']:.4f} vs Original: 4.00")
    
    calibrator_ex.plot_analysis()
    pricer_ex.plot_straddle_payoff(straddle_ex)