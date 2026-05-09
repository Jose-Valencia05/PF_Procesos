import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson

def calcular_tasas_lambda(df):
    """
    1. CÁLCULO DE TASAS (LAMBDA)
    Calcula el parámetro lambda (λ) por partido y por minuto para AST, REB, BLK y FG3M.
    
    JUSTIFICACIÓN MATEMÁTICA:
    En un Proceso de Poisson, el parámetro λ (lambda) representa la tasa media de 
    ocurrencia de un evento en un intervalo de tiempo o espacio fijo. 
    Dado que nuestros datos están a nivel de 'partido completo' (48 minutos en NBA), 
    nuestro intervalo base es 1 partido.
    - λ por partido = Media aritmética de los eventos observados por equipo.
    - λ por minuto = (λ por partido) / 48.
    """
    print("--- 1. CÁLCULO DE TASAS LAMBDA ---")
    columnas_analisis = ['AST', 'REB', 'BLK', 'FG3M']
    
    # Lambda por partido (intervalo = 48 minutos)
    lambdas_por_partido = df.groupby('TEAM_NAME')[columnas_analisis].mean()
    
    # Lambda por minuto (intervalo = 1 minuto)
    # Por las propiedades de los procesos de Poisson, si la tasa es constante,
    # el lambda de un sub-intervalo es proporcional al tiempo.
    lambdas_por_minuto = lambdas_por_partido / 48.0
    
    print("\nLambda promedio por partido (Muestra de 5 equipos):")
    print(lambdas_por_partido.head())
    
    print("\nLambda promedio por minuto (Muestra de 5 equipos):")
    print(lambdas_por_minuto.head())
    
    return lambdas_por_partido

def graficar_distribucion_poisson(df, lambdas_df, equipo_objetivo, variable):
    """
    2 y 3. DISTRIBUCIÓN TEÓRICA VS EMPÍRICA Y VISUALIZACIÓN
    Genera la PMF teórica y la superpone al histograma de probabilidad empírica.
    """
    # Filtramos los datos empíricos
    datos_equipo = df[df['TEAM_NAME'] == equipo_objetivo][variable]
    
    # Obtenemos nuestro λ previamente calculado
    lambda_param = lambdas_df.loc[equipo_objetivo, variable]
    
    # Definimos el dominio (k posibles eventos, desde 0 hasta un margen arriba del máximo)
    k_max = int(datos_equipo.max()) + 5
    k_values = np.arange(0, k_max)
    
    # Calculamos la PMF (Probability Mass Function): P(X = k) = (λ^k * e^-λ) / k!
    pmf_teorica = poisson.pmf(k_values, lambda_param)
    
    # -- Visualización --
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Datos Empíricos (Histograma)
    # Utilizamos stat='probability' para que el área/altura represente probabilidad (0 a 1)
    sns.histplot(datos_equipo, bins=np.arange(0, k_max+1)-0.5, stat='probability', 
                 color='skyblue', alpha=0.7, label=f'Empírica (Datos Reales)')
    
    # Modelo Teórico (Línea y Puntos)
    plt.plot(k_values, pmf_teorica, 'ro-', ms=6, label=f'Teórica Poisson (λ={lambda_param:.2f})')
    
    plt.title(f'Distribución de {variable} por partido - {equipo_objetivo}\n(Teórica vs Empírica)', fontsize=14)
    plt.xlabel(f'Número de {variable} (k)', fontsize=12)
    plt.ylabel('Probabilidad P(X = k)', fontsize=12)
    plt.xticks(np.arange(0, k_max, step=max(1, k_max//15)))
    plt.legend()
    
    # Guardamos la imagen
    filename = f'poisson_plot_{equipo_objetivo.replace(" ", "_")}_{variable}.png'
    plt.savefig(filename)
    print(f"Gráfico guardado en disco: {filename}")
    plt.close()

def ejecutar_analisis():
    # Cargar el dataset que generamos en el paso anterior
    try:
        df = pd.read_csv('dataset_nba_poisson_2025.csv')
        print(f"Dataset cargado correctamente. Total de partidos analizados: {len(df)}")
    except FileNotFoundError:
        print("Error: No se encontró 'dataset_nba_poisson_2025.csv'.")
        return

    # 1. Ejecutar cálculos matriciales
    lambdas_df = calcular_tasas_lambda(df)
    
    # 2. Extraer un equipo aleatorio (o el primero) para generar el caso de estudio visual
    equipo_estudio = df['TEAM_NAME'].iloc[0]
    
    print(f"\n--- 2. GENERANDO MODELOS VISUALES PARA: {equipo_estudio} ---")
    
    # Variables a contrastar para ver el comportamiento del supuesto de independencia
    variables_estudio = ['BLK', 'AST', 'REB']
    
    for var in variables_estudio:
        graficar_distribucion_poisson(df, lambdas_df, equipo_estudio, var)
        
    print("\n¡Análisis completado! Revisa las imágenes PNG generadas en tu carpeta.")

if __name__ == "__main__":
    ejecutar_analisis()
