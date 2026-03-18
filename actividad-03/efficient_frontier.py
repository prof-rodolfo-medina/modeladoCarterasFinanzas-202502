"""
Módulo: Frontera Eficiente y Optimización de Carteras
Universidad Internacional de La Rioja (UNIR)
Maestría en Ciencias Computacionales y Matemáticas Aplicadas

Autor: Dr. Rodolfo Rafael Medina Ramírez
Fecha: Agosto 2025
Versión: 1.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class EfficientFrontier:
    """
    Clase para calcular y visualizar la frontera eficiente de Markowitz.
    """
    
    def __init__(self, expected_returns, cov_matrix, risk_free_rate=0.02):
        """Inicializa la frontera eficiente."""
        if isinstance(expected_returns, pd.Series):
            expected_returns = expected_returns.values
        if isinstance(cov_matrix, pd.DataFrame):
            cov_matrix = cov_matrix.values
            
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(expected_returns)
        
    def portfolio_return(self, weights):
        """Calcula el rendimiento esperado de una cartera."""
        return np.dot(weights, self.expected_returns)
    
    def portfolio_volatility(self, weights):
        """Calcula la volatilidad (desviación estándar) de una cartera."""
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
    
    def sharpe_ratio(self, weights):
        """Calcula el ratio de Sharpe de una cartera."""
        port_return = self.portfolio_return(weights)
        port_volatility = self.portfolio_volatility(weights)
        return (port_return - self.risk_free_rate) / port_volatility
    
    def min_variance_portfolio(self):
        """Encuentra la cartera de mínima varianza."""
        objective = lambda w: np.dot(w.T, np.dot(self.cov_matrix, w))
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_guess = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(objective, initial_guess, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        weights = result.x
        
        return {
            'weights': weights,
            'return': self.portfolio_return(weights),
            'volatility': self.portfolio_volatility(weights),
            'sharpe': self.sharpe_ratio(weights)
        }
    
    def tangent_portfolio(self):
        """Encuentra la cartera tangente (maximiza ratio de Sharpe)."""
        def negative_sharpe(w):
            return -self.sharpe_ratio(w)
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_guess = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(negative_sharpe, initial_guess, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        weights = result.x
        
        return {
            'weights': weights,
            'return': self.portfolio_return(weights),
            'volatility': self.portfolio_volatility(weights),
            'sharpe': self.sharpe_ratio(weights)
        }
    
    def efficient_return(self, target_return):
        """Encuentra la cartera de mínima varianza para un rendimiento objetivo."""
        objective = lambda w: np.dot(w.T, np.dot(self.cov_matrix, w))
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: self.portfolio_return(w) - target_return}
        ]
        
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_guess = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(objective, initial_guess, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            weights = result.x
            return {
                'weights': weights,
                'return': self.portfolio_return(weights),
                'volatility': self.portfolio_volatility(weights),
                'sharpe': self.sharpe_ratio(weights)
            }
        return None
    
    def efficient_frontier_curve(self, n_points=50):
        """Genera puntos a lo largo de la frontera eficiente."""
        min_var = self.min_variance_portfolio()
        min_return = min_var['return']
        max_return = np.max(self.expected_returns)
        
        target_returns = np.linspace(min_return, max_return, n_points)
        
        returns, volatilities, sharpes = [], [], []
        
        for target in target_returns:
            result = self.efficient_return(target)
            if result is not None:
                returns.append(result['return'])
                volatilities.append(result['volatility'])
                sharpes.append(result['sharpe'])
        
        return np.array(returns), np.array(volatilities), np.array(sharpes)
    
    def random_portfolios(self, n_portfolios=10000):
        """Genera carteras aleatorias para visualización."""
        returns = np.zeros(n_portfolios)
        volatilities = np.zeros(n_portfolios)
        sharpes = np.zeros(n_portfolios)
        
        for i in range(n_portfolios):
            weights = np.random.random(self.n_assets)
            weights /= np.sum(weights)
            
            returns[i] = self.portfolio_return(weights)
            volatilities[i] = self.portfolio_volatility(weights)
            sharpes[i] = self.sharpe_ratio(weights)
        
        return returns, volatilities, sharpes
    
    def plot_efficient_frontier(self, show_random=True, n_random=10000, 
                               show_cml=True, figsize=(12, 8)):
        """Grafica la frontera eficiente completa."""
        plt.figure(figsize=figsize)
        
        if show_random:
            rand_returns, rand_vols, rand_sharpes = self.random_portfolios(n_random)
            scatter = plt.scatter(rand_vols * 100, rand_returns * 100, 
                                c=rand_sharpes, cmap='viridis', 
                                alpha=0.3, s=10, label='Carteras Aleatorias')
            plt.colorbar(scatter, label='Ratio de Sharpe')
        
        ef_returns, ef_vols, ef_sharpes = self.efficient_frontier_curve(100)
        plt.plot(ef_vols * 100, ef_returns * 100, 'b-', linewidth=3, 
                label='Frontera Eficiente')
        
        min_var = self.min_variance_portfolio()
        plt.scatter(min_var['volatility'] * 100, min_var['return'] * 100, 
                   marker='o', color='green', s=200, 
                   label='Mínima Varianza',
                   edgecolors='black', linewidth=2, zorder=5)
        
        tangent = self.tangent_portfolio()
        plt.scatter(tangent['volatility'] * 100, tangent['return'] * 100, 
                   marker='*', color='red', s=400, 
                   label=f"Cartera Tangente (Sharpe={tangent['sharpe']:.3f})",
                   edgecolors='black', linewidth=2, zorder=5)
        
        plt.scatter(0, self.risk_free_rate * 100, marker='D', 
                   color='blue', s=200, label=f'Rf = {self.risk_free_rate*100:.1f}%',
                   edgecolors='black', linewidth=2, zorder=5)
        
        if show_cml:
            max_vol = tangent['volatility'] * 1.5
            vol_range = np.linspace(0, max_vol, 100)
            cml_returns = self.risk_free_rate + (tangent['return'] - self.risk_free_rate) * vol_range / tangent['volatility']
            plt.plot(vol_range * 100, cml_returns * 100, 'r--', 
                    linewidth=2, label='CML', zorder=4)
        
        plt.xlabel('Volatilidad (%)', fontsize=12, fontweight='bold')
        plt.ylabel('Rendimiento Esperado (%)', fontsize=12, fontweight='bold')
        plt.title('Frontera Eficiente de Markowitz y CML', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()


if __name__ == "__main__":
    print("=" * 70)
    print("EJEMPLO: Frontera Eficiente y Optimización de Carteras")
    print("=" * 70)
    
    asset_names = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    expected_returns = np.array([0.15, 0.12, 0.13, 0.18, 0.20])
    
    corr_matrix = np.array([
        [1.00, 0.75, 0.80, 0.65, 0.55],
        [0.75, 1.00, 0.78, 0.70, 0.50],
        [0.80, 0.78, 1.00, 0.68, 0.52],
        [0.65, 0.70, 0.68, 1.00, 0.60],
        [0.55, 0.50, 0.52, 0.60, 1.00]
    ])
    
    volatilities = np.array([0.25, 0.22, 0.20, 0.30, 0.35])
    cov_matrix = np.outer(volatilities, volatilities) * corr_matrix
    
    ef = EfficientFrontier(expected_returns, cov_matrix, risk_free_rate=0.03)
    
    print("\n📊 CARTERA TANGENTE:")
    tangent = ef.tangent_portfolio()
    print(f"   Rendimiento: {tangent['return']*100:.2f}%")
    print(f"   Volatilidad: {tangent['volatility']*100:.2f}%")
    print(f"   Sharpe: {tangent['sharpe']:.4f}")
    
    print("\n📉 CARTERA MÍNIMA VARIANZA:")
    min_var = ef.min_variance_portfolio()
    print(f"   Rendimiento: {min_var['return']*100:.2f}%")
    print(f"   Volatilidad: {min_var['volatility']*100:.2f}%")
    
    print("\n" + "=" * 70)