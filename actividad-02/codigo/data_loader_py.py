"""
Data Loader para Actividad 02 - Algoritmo Hull-White
Universidad Internacional de La Rioja (UNIR)
Curso: Modelización y Valoración de Derivados y Carteras en Finanzas
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

class DataLoader:
    """Clase para cargar y generar datos de ejemplo para Hull-White"""
    
    def __init__(self, data_path: str = "datos/"):
        """
        Inicializa el cargador de datos.
        
        Args:
            data_path: Ruta a la carpeta de datos
        """
        self.data_path = Path(data_path)
    
    def load_sample_prices(self) -> pd.DataFrame:
        """
        Carga los precios de ejemplo desde CSV.
        
        Returns:
            DataFrame con columnas: fecha, precio, volumen, rendimiento
        """
        file_path = self.data_path / "precios_ejemplo.csv"
        
        if not file_path.exists():
            print(f"⚠️ Archivo no encontrado: {file_path}")
            print("💡 Generando datos sintéticos...")
            return self.generate_sample_prices()
        
        df = pd.read_csv(file_path)
        df['fecha'] = pd.to_datetime(df['fecha'])
        
        print(f"✅ Cargados {len(df)} precios de ejemplo")
        return df
    
    def load_straddle_data(self) -> pd.DataFrame:
        """
        Carga los datos del Long Straddle desde CSV.
        
        Returns:
            DataFrame con datos de opciones call y put
        """
        file_path = self.data_path / "datos_long_straddle.csv"
        
        if not file_path.exists():
            print(f"⚠️ Archivo no encontrado: {file_path}")
            print("💡 Generando datos sintéticos de straddle...")
            return self.generate_straddle_data()
        
        df = pd.read_csv(file_path)
        df['fecha'] = pd.to_datetime(df['fecha'])
        df['vencimiento'] = pd.to_datetime(df['vencimiento'])
        
        print(f"✅ Cargados {len(df)} registros de straddle")
        return df
    
    def generate_sample_prices(self, n_periods: int = 50, S0: float = 100.0,
                             mu: float = 0.001, sigma: float = 0.025) -> pd.DataFrame:
        """
        Genera serie sintética de precios para pruebas.
        
        Args:
            n_periods: Número de períodos
            S0: Precio inicial
            mu: Deriva esperada
            sigma: Volatilidad
            
        Returns:
            DataFrame con serie de precios sintética
        """
        np.random.seed(42)  # Para reproducibilidad
        
        # Generar fechas
        dates = pd.date_range(start='2024-01-02', periods=n_periods, freq='B')
        
        # Generar precios usando GBM discreto
        returns = np.random.normal(mu, sigma, n_periods)
        prices = [S0]
        
        for r in returns[:-1]:
            prices.append(prices[-1] * (1 + r))
        
        # Generar volúmenes correlacionados con volatilidad
        base_volume = 1500000
        volume_noise = np.random.normal(0, 0.2, n_periods)
        volumes = [int(base_volume * (1 + vn)) for vn in volume_noise]
        
        # Calcular rendimientos
        rendimientos = [np.nan] + [prices[i]/prices[i-1] - 1 for i in range(1, len(prices))]
        
        df = pd.DataFrame({
            'fecha': dates,
            'precio': prices,
            'volumen': volumes,
            'rendimiento': rendimientos
        })
        
        print(f"✅ Generados {len(df)} precios sintéticos")
        return df
    
    def generate_straddle_data(self, n_days: int = 24, S0: float = 100.0,
                             K: float = 100.0) -> pd.DataFrame:
        """
        Genera datos sintéticos de Long Straddle.
        
        Args:
            n_days: Número de días de datos
            S0: Precio inicial del subyacente
            K: Strike de las opciones
            
        Returns:
            DataFrame con datos de straddle
        """
        np.random.seed(123)
        
        # Generar fechas (días hábiles)
        dates = pd.date_range(start='2024-06-01', periods=n_days, freq='B')
        vencimiento = pd.Timestamp('2024-07-01')
        
        # Generar precios del subyacente
        returns = np.random.normal(0.002, 0.02, n_days)
        prices = [S0]
        for r in returns[:-1]:
            prices.append(prices[-1] * (1 + r))
        
        # Generar precios de opciones usando aproximación Black-Scholes simplificada
        r = 0.05  # Tasa libre de riesgo
        data = []
        
        for i, (date, S) in enumerate(zip(dates, prices)):
            # Tiempo hasta vencimiento (en años)
            T = (vencimiento - date).days / 365.25
            
            # Volatilidad implícita variable
            iv = 0.25 + 0.05 * np.random.normal()
            
            # Aproximación simple de precios de opciones
            # Call: valor intrínseco + valor temporal
            call_intrinsic = max(S - K, 0)
            call_time_value = max(0, 2 + 8 * T * iv + np.random.normal(0, 0.5))
            call_price = call_intrinsic + call_time_value
            
            # Put: valor intrínseco + valor temporal
            put_intrinsic = max(K - S, 0)
            put_time_value = max(0, 2 + 8 * T * iv + np.random.normal(0, 0.5))
            put_price = put_intrinsic + put_time_value
            
            # Volúmenes (mayor volumen cuando opciones están ATM)
            atm_factor = 1 / (1 + abs(S - K) / K)
            base_call_vol = 2500 * atm_factor
            base_put_vol = 2000 * atm_factor
            
            vol_call = int(base_call_vol * (1 + 0.3 * np.random.normal()))
            vol_put = int(base_put_vol * (1 + 0.3 * np.random.normal()))
            
            data.append({
                'fecha': date,
                'precio_subyacente': round(S, 2),
                'call_precio': round(call_price, 2),
                'put_precio': round(put_price, 2),
                'strike': K,
                'vencimiento': vencimiento,
                'volatilidad_implicita': round(iv, 4),
                'volumen_call': max(vol_call, 500),
                'volumen_put': max(vol_put, 400)
            })
        
        df = pd.DataFrame(data)
        print(f"✅ Generados {len(df)} registros de straddle sintéticos")
        return df
    
    def get_hull_white_test_data(self) -> Dict[str, List[float]]:
        """
        Proporciona datasets específicos para probar Hull-White.
        
        Returns:
            Diccionario con diferentes series de prueba
        """
        test_data = {
            'basico': [100, 105, 98, 103, 110, 95, 108, 102, 115],
            'straddle_original': [100, 102, 98, 105, 103, 99, 107, 104],
            'alta_volatilidad': [100, 150, 75, 120, 80, 160, 90, 140, 70],
            'baja_volatilidad': [100, 101, 99, 102, 98, 103, 97, 104, 96],
            'tendencia_alcista': [100, 105, 110, 115, 120, 125, 130, 135, 140],
            'tendencia_bajista': [100, 95, 90, 85, 80, 75, 70, 65, 60],
            'datos_reales_muestra': [100.0, 102.15, 98.75, 105.20, 103.45, 99.80, 
                                   107.35, 104.20, 101.85, 108.50, 106.75, 103.90]
        }
        
        print("✅ Datasets de prueba disponibles:")
        for nombre, datos in test_data.items():
            print(f"   • {nombre}: {len(datos)} observaciones")
        
        return test_data
    
    def save_data_to_csv(self, df: pd.DataFrame, filename: str) -> None:
        """
        Guarda DataFrame en archivo CSV.
        
        Args:
            df: DataFrame a guardar
            filename: Nombre del archivo
        """
        self.data_path.mkdir(exist_ok=True)
        file_path = self.data_path / filename
        df.to_csv(file_path, index=False)
        print(f"✅ Datos guardados en: {file_path}")
    
    def create_sample_files(self) -> None:
        """Crea archivos CSV de ejemplo si no existen."""
        
        # Crear carpeta de datos
        self.data_path.mkdir(exist_ok=True)
        
        # Generar y guardar precios de ejemplo
        sample_prices = self.generate_sample_prices()
        self.save_data_to_csv(sample_prices, "precios_ejemplo.csv")
        
        # Generar y guardar datos de straddle
        straddle_data = self.generate_straddle_data()
        self.save_data_to_csv(straddle_data, "datos_long_straddle.csv")
        
        print("🎉 Archivos CSV creados exitosamente!")


def main():
    """Función principal para demo."""
    print("🔄 Inicializando Data Loader...")
    
    loader = DataLoader()
    
    # Crear archivos si no existen
    loader.create_sample_files()
    
    # Cargar datos
    precios = loader.load_sample_prices()
    straddle = loader.load_straddle_data()
    test_data = loader.get_hull_white_test_data()
    
    # Mostrar resumen
    print("\n📊 Resumen de datos cargados:")
    print(f"   • Precios ejemplo: {len(precios)} registros")
    print(f"   • Datos straddle: {len(straddle)} registros") 
    print(f"   • Datasets de prueba: {len(test_data)} series")
    
    # Mostrar muestra de datos
    print("\n🔍 Muestra de precios ejemplo:")
    print(precios.head())
    
    print("\n🔍 Muestra de datos straddle:")
    print(straddle[['fecha', 'precio_subyacente', 'call_precio', 'put_precio']].head())


if __name__ == "__main__":
    main()