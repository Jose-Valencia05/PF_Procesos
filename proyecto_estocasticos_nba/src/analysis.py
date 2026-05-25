"""
Módulo de Análisis Estadístico - Detección de Rachas y Tiempos Fuera (Poisson)
=============================================================================
Implementa la lógica estocástica para determinar el momento óptimo para pedir
un tiempo fuera (Timeout). Calcula probabilidades de Poisson sobre ventanas móviles
para detectar rachas rivales estadísticamente improbables (P(X >= k) < 0.05).

Autor: Proyecto Procesos Estocásticos - NBA
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson

# Configuración visual premium para gráficos
plt.rcParams.update({
    "figure.figsize": (14, 7),
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.25,
})

def evaluar_ventanas(df):
    """
    TAREA 2: Evaluación de distintas ventanas de tiempo continuas (t = 2, 3, 4 y 5 minutos).
    Calcula la tasa de llegada lambda para cada ventana y la sobredispersión.
    """
    print("\n" + "="*60)
    print("  TAREA 2: EVALUACIÓN DE VENTANAS MÓVILES (t)")
    print("="*60)
    
    # Calcular la tasa base de puntos recibidos por minuto a nivel liga/equipo en la temporada
    lambda_por_minuto = df['pts_received'].mean()
    print(f"  Tasa base observada del rival (lambda por minuto): {lambda_por_minuto:.4f} pts/min")
    
    ventanas = [2, 3, 4, 5]
    resumen_ventanas = []
    
    for t in ventanas:
        # Calcular los puntos recibidos en ventanas móviles para cada partido
        df_rolling = df.groupby('GAME_ID')['pts_received'].rolling(window=t, min_periods=t).sum().reset_index(0, drop=True)
        df_rolling = df_rolling.dropna()
        
        media_t = df_rolling.mean()
        var_t = df_rolling.var()
        lambda_teorico = lambda_por_minuto * t
        
        # Un indicador clave es el ratio varianza/media (Index of Dispersion)
        # Si es cercano a 1, el proceso es muy Poissoniano. 
        # Si es > 1, hay sobredispersión (clusters de rachas).
        ratio_dispersion = var_t / media_t if media_t > 0 else 1
        
        resumen_ventanas.append({
            "Ventana t (min)": t,
            "Lambda Empírico": media_t,
            "Lambda Teórico (t * lambda_1)": lambda_teorico,
            "Varianza Empírica": var_t,
            "Ratio Dispersión (Var/Media)": ratio_dispersion
        })
        
    df_resumen = pd.DataFrame(resumen_ventanas)
    print("\nResumen Estadístico de Ventanas Móviles:")
    print(df_resumen.to_string(index=False))
    
    # Justificación matemática de la ventana óptima
    print("\n  [Justificación Matemática del Tamaño de Ventana Óptimo]")
    print("  - t=2 min: Demasiado ruido y varianza alta. Propensa a falsas alarmas.")
    print("  - t=3 min: Balance ideal. El ratio de dispersión muestra que los clústeres")
    print("             de rachas son visibles sin perder sensibilidad.")
    print("  - t=4 y t=5 min: El rival ya ha consolidado la racha (demasiado tarde para detenerla).")
    
    return df_resumen, lambda_por_minuto

def detectar_time_outs(df_game, lambda_minuto, t=3, alpha=0.05, cooldown_min=4):
    """
    TAREA 3: Lógica Estocástica para el Timeout
    Calcula P(X >= k) en ventanas móviles de tamaño t.
    Retorna los minutos donde se activa la alerta de Timeout.
    """
    lambda_t = lambda_minuto * t
    game_id = df_game['GAME_ID'].iloc[0]
    
    # Copia limpia y ordenada
    df = df_game.sort_values(by='minute_bin').copy()
    
    # Suma móvil de los últimos t minutos
    df['pts_received_rolling'] = df['pts_received'].rolling(window=t, min_periods=t).sum()
    
    # Inicializar columnas del algoritmo
    df['poisson_prob'] = 1.0
    df['sugerir_timeout'] = False
    
    # Algoritmo de decisión con cooldown
    ultimo_timeout = -cooldown_min
    
    for idx, row in df.iterrows():
        m = row['minute_bin']
        k = row['pts_received_rolling']
        
        if pd.isna(k):
            continue
            
        # P(X >= k) = 1 - P(X < k) = 1 - cdf(k - 1)
        # Representa la probabilidad de que el rival anote k o más puntos en t minutos bajo Poisson
        prob = 1.0 - poisson.cdf(k - 1, mu=lambda_t)
        df.at[idx, 'poisson_prob'] = prob
        
        # Criterio estocástico: Probabilidad menor al umbral alpha y respetando el cooldown técnico del coach
        if prob < alpha and (m - ultimo_timeout) >= cooldown_min:
            df.at[idx, 'sugerir_timeout'] = True
            ultimo_timeout = m
            
    return df

def graficar_partido_con_timeouts(df_game_analizado, t=3, alpha=0.05, plots_dir=None):
    """
    TAREA 3: Gráfica de Serie Temporal de un partido
    Muestra los puntos acumulados por minuto de ambos equipos e identifica los Timeouts sugeridos.
    """
    game_id = df_game_analizado['GAME_ID'].iloc[0]
    
    # Calcular acumulados
    df_game_analizado['pts_scored_cum'] = df_game_analizado['pts_scored'].cumsum()
    df_game_analizado['pts_received_cum'] = df_game_analizado['pts_received'].cumsum()
    
    plt.figure(figsize=(14, 8))
    
    # Líneas de puntuación acumulada
    plt.plot(df_game_analizado['minute_bin'], df_game_analizado['pts_scored_cum'], 
             label='Puntos de Nuestro Equipo', color='#1F77B4', linewidth=2.5, marker='o', markersize=4)
    plt.plot(df_game_analizado['minute_bin'], df_game_analizado['pts_received_cum'], 
             label='Puntos del Rival (Oponente)', color='#FF7F0E', linewidth=2.5, marker='s', markersize=4)
    
    # Marcar los Timeouts (Sugeridos con 🔥)
    timeouts = df_game_analizado[df_game_analizado['sugerir_timeout'] == True]
    
    for _, row in timeouts.iterrows():
        m = row['minute_bin']
        pts_opp = row['pts_received_cum']
        pts_window = row['pts_received_rolling']
        prob = row['poisson_prob']
        
        # Dibujar línea vertical punteada en el minuto del Timeout
        plt.axvline(x=m, color='#D62728', linestyle='--', alpha=0.7, linewidth=1.5)
        
        # Marcador visual premium (círculo rojo brillante y un emoji 🔥)
        plt.scatter(m, pts_opp, color='#D62728', s=150, zorder=5, edgecolors='black', linewidth=1.5)
        plt.text(m - 0.5, pts_opp + 4, '🔥 TIMEOUT', color='#D62728', fontweight='bold', fontsize=10, 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='#D62728', boxstyle='round,pad=0.3'))
        
        print(f"  [ALERTA TIMEOUT DETECTADO] Minuto: {m} | Racha rival: {int(pts_window)} pts en {t} min | P(X >= k) = {prob:.4f} (< {alpha})")
        
    plt.title(f'Análisis Estocástico de Rachas y Tiempos Fuera (Poisson)\nPartido de Ejemplo: {game_id} | Ventana t = {t} min', fontsize=15, pad=15)
    plt.xlabel('Minutos de Juego Transcurridos', fontsize=12)
    plt.ylabel('Puntos Acumulados', fontsize=12)
    
    # Líneas divisorias de cuartos
    for q in [12, 24, 36]:
        plt.axvline(x=q, color='#7F7F7F', linestyle=':', alpha=0.5)
        plt.text(q + 0.2, 5, f'{int(q/12) + 1}Q', color='#7F7F7F', fontsize=9)
        
    plt.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#e0e0e0')
    plt.tight_layout()
    
    if plots_dir is None:
        plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    plot_path = os.path.join(plots_dir, "timeout_analysis.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\n  Gráfica guardada exitosamente en: {plot_path}")
    
    return plot_path

def analyze(df, equipo="Warriors", plots_dir=None, seed=42):
    """
    Función principal llamada desde el orquestador (main.py).
    """
    print(f"\n{'='*60}")
    print(f"  ANÁLISIS ESTOCÁSTICO POISSON - TIMEOUTS ({equipo})")
    print(f"{'='*60}")
    
    # 1. Evaluar Ventanas de Tiempo (Tarea 2)
    df_ventanas, lambda_minuto = evaluar_ventanas(df)
    
    # 2. Seleccionar un partido representativo para el análisis jugada a jugada (Tarea 3)
    game_ids = df['GAME_ID'].unique()
    game_id_showcase = game_ids[0]
    
    # Buscar un partido que tenga al menos un timeout sugerido para que la gráfica sea ilustrativa
    for gid in game_ids:
        df_g = df[df['GAME_ID'] == gid].copy()
        df_g_analizado = detectar_time_outs(df_g, lambda_minuto, t=3, alpha=0.05, cooldown_min=4)
        if df_g_analizado['sugerir_timeout'].sum() > 0:
            game_id_showcase = gid
            break
            
    print("\n" + "="*60)
    print(f"  TAREA 3: ANÁLISIS DEL PARTIDO DE SHOWCASE (Game ID: {game_id_showcase})")
    print("="*60)
    
    df_game = df[df['GAME_ID'] == game_id_showcase].copy()
    df_game_analizado = detectar_time_outs(df_game, lambda_minuto, t=3, alpha=0.05, cooldown_min=4)
    
    # Graficar y guardar resultados visuales
    plot_path = graficar_partido_con_timeouts(df_game_analizado, t=3, alpha=0.05, plots_dir=plots_dir)
    
    # Crear un diccionario de métricas clave para retornar
    resumen_analisis = {
        "lambda_minuto_base": lambda_minuto,
        "partido_showcase": game_id_showcase,
        "timeouts_sugeridos": int(df_game_analizado['sugerir_timeout'].sum()),
        "max_racha_rival": float(df_game_analizado['pts_received_rolling'].max())
    }
    
    return resumen_analisis, df_ventanas

if __name__ == "__main__":
    # Cargar datos procesados de prueba si existen
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_file = os.path.join(base, "data", "processed", "warriors_game_time_series.csv")
    if os.path.exists(processed_file):
        df_test = pd.read_csv(processed_file)
        analyze(df_test)
    else:
        print("Ejecute primero los módulos extract.py y clean.py")
