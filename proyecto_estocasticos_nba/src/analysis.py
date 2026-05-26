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
    Realiza la prueba de Kolmogorov-Smirnov para una distribución de Poisson.

    Dado que el parámetro lambda se estima de los datos, se utiliza una
    simulación Monte Carlo para obtener el p-valor corregido.

    Parameters
    ----------
    puntos : array-like
        Serie de puntos anotados por partido.
    lambda_est : float
        Parámetro lambda estimado.
    alpha : float
        Nivel de significancia.
    n_sim : int
        Número de simulaciones para el p-valor Monte Carlo.

    Returns
    -------
    dict
        Diccionario con estadístico KS, p-valor (estándar y MC), y conclusión.
    """
    p = np.array(puntos)
    n = len(p)

    ks_stat, p_valor_std = stats.kstest(p, lambda x: poisson.cdf(x, mu=lambda_est))

    ks_sim = np.zeros(n_sim)
    for i in range(n_sim):
        sim_sample = np.random.poisson(lam=lambda_est, size=n)
        lambda_sim = np.mean(sim_sample)
        ks_sim[i], _ = stats.kstest(sim_sample, lambda x: poisson.cdf(x, mu=lambda_sim))

    p_valor_mc = np.mean(ks_sim >= ks_stat)
    rechazar = p_valor_mc < alpha

    if rechazar:
        conclusion = (
            f"Se RECHAZA H0 (p_MC={p_valor_mc:.4f} < alpha={alpha}). "
            f"Los datos NO siguen una distribución de Poisson."
        )
    else:
        conclusion = (
            f"NO se rechaza H0 (p_MC={p_valor_mc:.4f} >= alpha={alpha}). "
            f"Los datos son consistentes con una distribución de Poisson."
        )

    return {
        "estadistico_ks": ks_stat,
        "p_valor_estandar": p_valor_std,
        "p_valor_monte_carlo": p_valor_mc,
        "n_simulaciones": n_sim,
        "conclusion": conclusion,
        "nivel_significancia": alpha,
        "rechazar_h0": rechazar,
    }


def simular_poisson(lambda_est, n, seed=42):
    """
    Simula datos de una distribución de Poisson.

    Parameters
    ----------
    lambda_est : float
        Parámetro lambda.
    n : int
        Número de simulaciones (partidos).
    seed : int
        Semilla para reproducibilidad.

    Returns
    -------
    numpy.ndarray
        Array con datos simulados.
    """
    rng = np.random.default_rng(seed)
    return rng.poisson(lam=lambda_est, size=n)


def graficar_histograma_comparativo(puntos_reales, lambda_est, puntos_simulados=None,
                                     equipo="Equipo", plots_dir=None):
    """
    Genera un histograma comparativo: datos reales + PMF teórica + simulación.

    Parameters
    ----------
    puntos_reales : array-like
        Puntos reales por partido.
    lambda_est : float
        Lambda estimado.
    puntos_simulados : array-like
        Puntos simulados de una Poisson(lambda_est).
    equipo : str
        Nombre del equipo para el título.
    plots_dir : str, optional
        Directorio donde guardar la gráfica.

    Returns
    -------
    str
        Ruta del archivo guardado.
    """
    ncols = 2 if puntos_simulados is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(16 if ncols == 2 else 10, 6))
    if ncols == 1:
        axes = [axes]

    # --- Panel izquierdo: Reales + PMF teórica ---
    ax = axes[0]
    p_real = np.array(puntos_reales).astype(int)
    min_val = min(p_real.min(), int(lambda_est - 4 * np.sqrt(lambda_est)))
    max_val = max(p_real.max(), int(lambda_est + 4 * np.sqrt(lambda_est)))
    bins = np.arange(min_val - 0.5, max_val + 1.5, 1)

    ax.hist(p_real, bins=bins, density=True, alpha=0.65, color="#2196F3",
            edgecolor="white", linewidth=0.5, label="Frecuencia real")

    x_pmf, y_pmf = poisson_pmf_teorica(lambda_est, x_min=min_val, x_max=max_val)
    ax.plot(x_pmf, y_pmf, "o-", color="#D32F2F", linewidth=2, markersize=4,
            label=f"PMF Poisson(λ={lambda_est:.1f})")

    ax.axvline(lambda_est, color="#D32F2F", linestyle="--", alpha=0.5,
               label=f"Media = {lambda_est:.1f}")
    ax.set_xlabel("Puntos por partido")
    ax.set_ylabel("Densidad de probabilidad")
    ax.set_title(f"Datos Reales vs Distribución Poisson Teórica")
    ax.legend(loc="upper right")
    ax.yaxis.set_major_locator(MaxNLocator(integer=False))

    ax.text(0.98, 0.95,
            f"λ estimado = {lambda_est:.2f}\n"
            f"n = {len(p_real)} partidos\n"
            f"Media real = {np.mean(p_real):.2f}\n"
            f"Varianza real = {np.var(p_real):.2f}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    # --- Panel derecho: solo si hay simulacion ---
    if puntos_simulados is not None:
        ax = axes[1]
        p_sim = np.array(puntos_simulados).astype(int)
        both = np.concatenate([p_real, p_sim])
        min_all = both.min()
        max_all = both.max()
        bins_all = np.arange(min_all - 0.5, max_all + 1.5, 1)

        ax.hist(p_real, bins=bins_all, density=True, alpha=0.55, color="#2196F3",
                edgecolor="white", linewidth=0.5, label="Frecuencia real")
        ax.hist(p_sim, bins=bins_all, density=True, alpha=0.45, color="#FF9800",
                edgecolor="white", linewidth=0.5, label=f"Simulacion (n={len(p_sim)})")

        ax.set_xlabel("Puntos por partido")
        ax.set_ylabel("Densidad de probabilidad")
        ax.set_title(f"Datos Reales vs Simulacion Monte Carlo")
        ax.legend(loc="upper right")

        ax.text(0.98, 0.95,
                "La simulacion genera partidos sinteticos\n"
                "usando Poisson(lambda). Si las barras se parecen,\n"
                "el modelo es adecuado.",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    plt.tight_layout()

    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
    filepath = os.path.join(plots_dir, "histograma_comparativo.png") if plots_dir else None
    if filepath:
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Gráfica guardada: {filepath}")
    else:
        plt.close(fig)

    return filepath


def graficar_qq_poisson(puntos_reales, lambda_est, equipo="Equipo", plots_dir=None):
    """
    Genera un Q-Q plot para evaluar visualmente el ajuste a la distribución
    de Poisson.

    Parameters
    ----------
    puntos_reales : array-like
        Puntos reales por partido.
    lambda_est : float
        Lambda estimado.
    equipo : str
        Nombre del equipo.
    plots_dir : str, optional
        Directorio donde guardar la gráfica.

    Returns
    -------
    str
        Ruta del archivo guardado.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    p_real = np.array(puntos_reales)
    n = len(p_real)
    p_sorted = np.sort(p_real)
    theoretical_quantiles = poisson.ppf((np.arange(1, n + 1) - 0.5) / n, mu=lambda_est)

    ax.scatter(theoretical_quantiles, p_sorted, alpha=0.5, color="#2196F3",
               edgecolors="white", linewidth=0.3, s=50)
    ax.plot(theoretical_quantiles, theoretical_quantiles, "--", color="#D32F2F",
            linewidth=2, label="Línea de referencia (y = x)")

    ax.set_xlabel("Cuantiles teóricos (Poisson)")
    ax.set_ylabel("Cuantiles observados")
    ax.set_title(f"Q-Q Plot: Cuantiles Observados vs Cuantiles Teóricos Poisson")
    ax.legend(loc="upper left")
    ax.set_aspect("equal")

    ax.text(0.02, 0.95,
            "Si los puntos caen sobre la línea roja y=x,\n"
            "los datos se distribuyen como una Poisson.\n"
            "Desviaciones en las colas indican diferencias.",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax.text(0.98, 0.10,
            f"λ = {lambda_est:.2f}\n"
            f"n = {n} partidos",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
    filepath = os.path.join(plots_dir, "qq_plot_poisson.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gráfica guardada: {filepath}")
    return filepath


def graficar_serie_temporal(df, equipo="Equipo", plots_dir=None):
    """
    Grafica los puntos anotados por partido en orden cronológico para
    evaluar estacionariedad de la tasa lambda.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con columnas 'fecha' y 'puntos'.
    equipo : str
        Nombre del equipo.
    plots_dir : str
        Directorio para guardar la gráfica.

    Returns
    -------
    str
        Ruta del archivo guardado.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    fecha_col = "fecha" if "fecha" in df.columns else "GAME_DATE"
    pts_col = "puntos" if "puntos" in df.columns else "PTS"

    media_global = df[pts_col].mean()

    ax.plot(df[fecha_col], df[pts_col], "o-", markersize=3, linewidth=0.8,
            color="#2196F3", alpha=0.7)
    ax.axhline(media_global, color="#D32F2F", linestyle="--", linewidth=1.5,
               label=f"Media global = {media_global:.1f}")

    ax.set_xlabel("Fecha del partido")
    ax.set_ylabel("Puntos anotados")
    ax.set_title(f"Serie Temporal de Puntos por Partido — {equipo}")
    ax.legend(loc="upper right")

    plt.tight_layout()

    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
    filepath = os.path.join(plots_dir, "serie_temporal_puntos.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gráfica guardada: {filepath}")
    return filepath


def graficar_cdf_escalonada(puntos_reales, lambda_est, equipo="Equipo",
                            plots_dir=None):
    """
    Genera la CDF empírica escalonada vs la CDF teórica de Poisson.

    La línea roja punteada es la CDF teórica de Poisson(lambda).
    Los escalones azules son la CDF empírica de los datos reales.
    Mientras más se traslapen, mejor es el ajuste.

    Parameters
    ----------
    puntos_reales : array-like
        Puntos reales por partido.
    lambda_est : float
        Lambda estimado de la distribución de Poisson.
    equipo : str
        Nombre del equipo para el título.
    plots_dir : str, optional
        Directorio donde guardar la gráfica.

    Returns
    -------
    str
        Ruta del archivo guardado.
    """
    p_real = np.array(puntos_reales)
    n = len(p_real)
    p_sorted = np.sort(p_real)

    cdf_empirica = np.arange(1, n + 1) / n

    x_min = p_sorted[0]
    x_max = p_sorted[-1]
    x_theo = np.arange(x_min, x_max + 1)
    cdf_teorica = poisson.cdf(x_theo, mu=lambda_est)

    max_diff = np.max(np.abs(
        np.searchsorted(p_sorted, x_theo, side="right") / n - cdf_teorica
    ))

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.step(p_sorted, cdf_empirica, where="post", color="#2196F3",
             linewidth=2, label="CDF empírica (datos reales)")

    ax.plot(x_theo, cdf_teorica, "o--", color="#D32F2F", linewidth=2,
             markersize=5, label=f"CDF teórica Poisson (λ={lambda_est:.1f})")

    ax.set_xlabel("Puntos por partido")
    ax.set_ylabel("Probabilidad acumulada")
    ax.set_title(
        f"Función de Distribución Acumulada (CDF) — {equipo}"
    )
    ax.legend(loc="lower right")

    ax.text(0.02, 0.95,
            "La línea roja punteada es la CDF teórica de Poisson(λ).\n"
            "Los escalones azules son la CDF empírica de los datos reales.\n"
            "Mientras más se traslapen, mejor es el ajuste.",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax.text(0.98, 0.20,
            f"λ = {lambda_est:.2f}\n"
            f"n = {n} partidos\n"
            f"Diferencia máxima = {max_diff:.4f}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()

    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
    filepath = os.path.join(plots_dir, "cdf_poisson.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gráfica guardada: {filepath}")
    return filepath


def graficar_barras_chi2(resultado_chi2, lambda_est, n_total,
                          equipo="Equipo", plots_dir=None):
    """
    Grafica las frecuencias observadas vs esperadas usadas en la prueba
    chi-cuadrada.

    Parameters
    ----------
    resultado_chi2 : dict
        Resultado de prueba_chi_cuadrada().
    lambda_est : float
        Lambda estimado.
    n_total : int
        Número total de partidos.
    equipo : str
        Nombre del equipo.
    plots_dir : str
        Directorio para guardar la gráfica.

    Returns
    -------
    str
        Ruta del archivo guardado.
    """
    if resultado_chi2.get("grados_libertad") is None or np.isnan(resultado_chi2.get("grados_libertad", np.nan)):
        print("  No se generó gráfica chi2: prueba no válida.")
        return None

    labels = resultado_chi2["bins_labels"]
    obs = resultado_chi2["observados"]
    esp = resultado_chi2["esperados"]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width / 2, obs, width, label="Observadas",
                   color="#2196F3", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, esp, width, label="Esperadas (Poisson)",
                   color="#FF9800", edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Rango de puntos por partido")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"Frecuencias Observadas vs Esperadas (lambda={lambda_est:.1f}, n={n_total}) — {equipo}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.legend(loc="upper right")

    ax.text(0.98, 0.95,
            f"chi2 = {resultado_chi2['estadistico_chi2']:.2f}\n"
            f"gl = {resultado_chi2['grados_libertad']}\n"
            f"p = {resultado_chi2['p_valor']:.4f}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()

    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
    filepath = os.path.join(plots_dir, "barras_chi2.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gráfica guardada: {filepath}")
    return filepath


def graficar_media_varianza_por_temporada(df, equipo="Equipo", plots_dir=None):
    """
    Calcula y grafica la media y varianza de puntos por temporada para
    evaluar la estabilidad de lambda a lo largo del tiempo.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con columnas 'fecha' y 'puntos'.
    equipo : str
        Nombre del equipo.
    plots_dir : str
        Directorio para guardar la gráfica.

    Returns
    -------
    str
        Ruta del archivo guardado.
    """
    fecha_col = "fecha" if "fecha" in df.columns else "GAME_DATE"
    pts_col = "puntos" if "puntos" in df.columns else "PTS"

    df_temp = df.copy()
    df_temp["temporada"] = df_temp[fecha_col].apply(
        lambda d: f"{d.year}-{str(d.year + 1)[-2:]}" if d.month >= 10
        else f"{d.year - 1}-{str(d.year)[-2:]}"
    )

    agg = df_temp.groupby("temporada")[pts_col].agg(["mean", "var", "count"])
    agg = agg[agg["count"] >= 10]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(agg))

    ax.plot(x, agg["mean"], "o-", color="#2196F3", linewidth=2, markersize=8,
            label="Media por temporada")
    ax.plot(x, agg["var"], "s--", color="#FF9800", linewidth=2, markersize=8,
            label="Varianza por temporada")
    ax.axhline(df[pts_col].mean(), color="#D32F2F", linestyle=":", linewidth=1.5,
               label=f"Media global = {df[pts_col].mean():.1f}")

    ax.set_xticks(x)
    ax.set_xticklabels(agg.index, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Temporada")
    ax.set_ylabel("Puntos")
    ax.set_title(f"Estabilidad de λ por Temporada — {equipo}")
    ax.legend(loc="upper left")

    ratio_global = df[pts_col].var() / df[pts_col].mean()

    ax.text(0.98, 0.95,
            "En una Poisson, E[X] = Var(X) = λ.\n"
            "Este gráfico verifica si λ se mantiene estable\n"
            "a lo largo de las temporadas. Las líneas deben\n"
            "estar cerca y ser horizontales.",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax.text(0.02, 0.20,
            f"Media global = {df[pts_col].mean():.2f}\n"
            f"Varianza global = {df[pts_col].var():.2f}\n"
            f"Razón Var/Media = {ratio_global:.4f}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()

    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
    filepath = os.path.join(plots_dir, "media_varianza_por_temporada.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gráfica guardada: {filepath}")
    return filepath


def identificar_eventos_y_tiempos(puntos, lambda_est, factor=1.0):
    """
    Define un 'evento' como un partido donde los puntos superan
    lambda + factor * desviacion. Calcula los tiempos entre eventos
    consecutivos.

    Parameters
    ----------
    puntos : array-like
        Puntos por partido.
    lambda_est : float
        Lambda estimado (media).
    factor : float
        Multiplicador de la desviacion para definir el umbral.

    Returns
    -------
    tuple
        (tiempos_entre_eventos, umbral, n_eventos)
    """
    p = np.array(puntos)
    desv = np.std(p)
    umbral = lambda_est + factor * desv

    indices_eventos = np.where(p > umbral)[0]

    if len(indices_eventos) < 2:
        return np.array([]), umbral, len(indices_eventos)

    tiempos = np.diff(indices_eventos)
    return tiempos, umbral, len(indices_eventos)


def graficar_qq_exponencial(tiempos_entre_eventos, lambda_exp, umbral,
                            equipo="Equipo", plots_dir=None):
    """
    Genera un Q-Q plot para evaluar si los tiempos entre eventos
    siguen una distribucion exponencial.

    Parameters
    ----------
    tiempos_entre_eventos : array-like
        Tiempos (en partidos) entre eventos consecutivos.
    lambda_exp : float
        Parametro lambda de la exponencial (1/media).
    umbral : float
        Umbral usado para definir un evento.
    equipo : str
        Nombre del equipo.
    plots_dir : str, optional
        Directorio donde guardar la grafica.

    Returns
    -------
    str
        Ruta del archivo guardado.
    """
    if len(tiempos_entre_eventos) < 5:
        print("  Muy pocos eventos para Q-Q exponencial. Se omite.")
        return None

    t = np.array(tiempos_entre_eventos)
    n_t = len(t)
    t_sorted = np.sort(t)
    theoretical = stats.expon.ppf((np.arange(1, n_t + 1) - 0.5) / n_t,
                                   scale=1.0 / lambda_exp)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(theoretical, t_sorted, alpha=0.5, color="#4CAF50",
               edgecolors="white", linewidth=0.3, s=50)
    ax.plot(theoretical, theoretical, "--", color="#D32F2F",
            linewidth=2, label="Linea de referencia (y = x)")

    ax.set_xlabel("Cuantiles teoricos (Exponencial)")
    ax.set_ylabel("Cuantiles observados")
    ax.set_title(
        "Q-Q Plot Exponencial: Tiempos entre Eventos de Altos Puntos"
    )
    ax.legend(loc="upper left")

    ax.text(0.02, 0.95,
            "Un evento es un partido donde los puntos\n"
            f"superan el umbral = media + desviacion = {umbral:.1f}.\n"
            "Si los puntos caen sobre la linea y=x,\n"
            "los tiempos entre eventos siguen una Exponencial.",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax.text(0.98, 0.10,
            f"lambda_exp = {lambda_exp:.4f}\n"
            f"media obs = {np.mean(t):.2f} partidos\n"
            f"n eventos = {n_t}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()

    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
    filepath = os.path.join(plots_dir, "qq_exponencial.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grafica guardada: {filepath}")
    return filepath


def prueba_estacionariedad(puntos):
    """
    Realiza pruebas formales de estacionariedad sobre la serie de puntos:
    - ADF (Augmented Dickey-Fuller): H0 = no estacionaria (raiz unitaria)
    - KPSS: H0 = estacionaria

    Parameters
    ----------
    puntos : array-like
        Serie de puntos por partido.

    Returns
    -------
    dict
        Resultados de las pruebas ADF y KPSS.
    """
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
    except ImportError:
        return {
            "adf_estadistico": None,
            "adf_p_valor": None,
            "adf_conclusion": "statsmodels no instalado.",
            "kpss_estadistico": None,
            "kpss_p_valor": None,
            "kpss_conclusion": "statsmodels no instalado.",
        }

    p = np.array(puntos, dtype=float)

    adf_result = adfuller(p, autolag="AIC")
    adf_stat = adf_result[0]
    adf_p = adf_result[1]
    adf_conc = (
        "Serie ESTACIONARIA (rechaza H0 de raiz unitaria)"
        if adf_p < 0.05 else
        "Serie NO estacionaria (no rechaza H0)"
    )

    kpss_result = kpss(p, regression="c", nlags="auto")
    kpss_stat = kpss_result[0]
    kpss_p = kpss_result[1]
    kpss_conc = (
        "Serie ESTACIONARIA (no rechaza H0)"
        if kpss_p >= 0.05 else
        "Serie NO estacionaria (rechaza H0 de estacionariedad)"
    )

    return {
        "adf_estadistico": adf_stat,
        "adf_p_valor": adf_p,
        "adf_conclusion": adf_conc,
        "kpss_estadistico": kpss_stat,
        "kpss_p_valor": kpss_p,
        "kpss_conclusion": kpss_conc,
    }


def analizar_poisson_no_homogeneo(df):
    """
    Compara el modelo Poisson homogeneo (lambda unico)
    vs no homogeneo (lambda varia por local/visitante).

    Calcula lambda para partidos en casa (vs.) y fuera (@)
    y evalua cual modelo ajusta mejor via AIC.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con columnas 'enfrentamiento' y 'puntos'.

    Returns
    -------
    dict
        Resultados de la comparacion.
    """
    pts_col = "puntos" if "puntos" in df.columns else "PTS"
    matchup_col = "enfrentamiento" if "enfrentamiento" in df.columns else "MATCHUP"
    puntos = df[pts_col].values.astype(float)

    if matchup_col not in df.columns:
        return {
            "lambda_global": np.mean(puntos),
            "lambda_local": None,
            "lambda_visitante": None,
            "aic_homogeneo": None,
            "aic_no_homogeneo": None,
            "conclusion": "Sin datos de local/visitante.",
        }

    mask_local = df[matchup_col].str.contains("vs\\.", case=False, na=False)
    mask_visitante = df[matchup_col].str.contains("@", case=False, na=False)

    lambda_global = np.mean(puntos)
    lambda_local = np.mean(puntos[mask_local]) if mask_local.sum() > 0 else lambda_global
    lambda_visitante = np.mean(puntos[mask_visitante]) if mask_visitante.sum() > 0 else lambda_global

    ll_homogeneo = np.sum(poisson.logpmf(puntos, mu=lambda_global))
    k_hom = 1
    aic_hom = 2 * k_hom - 2 * ll_homogeneo

    ll_no_hom = 0
    k_no_hom = 2

    if mask_local.sum() > 0:
        ll_no_hom += np.sum(poisson.logpmf(puntos[mask_local], mu=lambda_local))
    if mask_visitante.sum() > 0:
        ll_no_hom += np.sum(poisson.logpmf(puntos[mask_visitante], mu=lambda_visitante))

    aic_no_hom = 2 * k_no_hom - 2 * ll_no_hom

    if aic_hom < aic_no_hom:
        conclusion = (
            "El modelo HOMOGENEO tiene mejor AIC. "
            "No hay evidencia suficiente de que lambda varie por local/visitante."
        )
    else:
        conclusion = (
            "El modelo NO HOMOGENEO tiene mejor AIC. "
            "Lambda varia significativamente entre partidos de local y visitante."
        )

    return {
        "lambda_global": lambda_global,
        "lambda_local": lambda_local,
        "lambda_visitante": lambda_visitante,
        "n_local": int(mask_local.sum()),
        "n_visitante": int(mask_visitante.sum()),
        "aic_homogeneo": aic_hom,
        "aic_no_homogeneo": aic_no_hom,
        "conclusion": conclusion,
    }


def resumen_estadistico(stats_dict, propiedad, estacionariedad=None,
                       no_homogeneo=None):
    """
    Genera un resumen de todos los resultados estadisticos en formato
    de diccionario, listo para mostrar o exportar.

    Parameters
    ----------
    stats_dict : dict
        Resultado de estadistica_descriptiva().
    propiedad : dict
        Resultado de verificar_propiedad_poisson().
    estacionariedad : dict, optional
        Resultado de prueba_estacionariedad().
    no_homogeneo : dict, optional
        Resultado de analizar_poisson_no_homogeneo().

    Returns
    -------
    dict
        Resumen completo del analisis.
    """
    result = {
        "estadistica_descriptiva": stats_dict,
        "propiedad_poisson": propiedad,
    }
    if estacionariedad is not None:
        result["estacionariedad"] = estacionariedad
    if no_homogeneo is not None:
        result["poisson_no_homogeneo"] = no_homogeneo
    return result


def imprimir_resumen(resumen):
    """
    Imprime en consola un resumen formateado de todos los resultados.

    Parameters
    ----------
    resumen : dict
        Resultado de resumen_estadistico().
    """
    sd = resumen["estadistica_descriptiva"]
    pp = resumen["propiedad_poisson"]

    print(f"\n{'='*60}")
    print(f"  RESUMEN DEL ANALISIS ESTADISTICO")
    print(f"{'='*60}")

    print(f"\n  --- Estadistica Descriptiva ---")
    print(f"  Partidos analizados (n):    {sd['n']}")
    print(f"  Media muestral (lambda):    {sd['media']:.2f}")
    print(f"  Varianza muestral:          {sd['varianza']:.2f}")
    print(f"  Desviacion estandar:        {sd['desviacion_estandar']:.2f}")
    print(f"  Minimo:                     {sd['minimo']:.0f}")
    print(f"  Maximo:                     {sd['maximo']:.0f}")
    print(f"  Asimetria:                  {sd['asimetria']:.3f}")
    print(f"  Curtosis:                   {sd['curtosis']:.3f}")

    print(f"\n  --- Propiedad E[X] = Var(X) = lambda ---")
    print(f"  Razon Var(X)/E[X]:          {pp['ratio_varianza_media']:.4f}")
    print(f"  Diferencia relativa:        {pp['diferencia_relativa']:.4f}")
    print(f"  Diagnostico:                {pp['diagnostico']}")

    est = resumen.get("estacionariedad")
    if est and est.get("adf_estadistico") is not None:
        print(f"\n  --- Pruebas de Estacionariedad ---")
        print(f"  ADF estadistico:            {est['adf_estadistico']:.4f}")
        print(f"  ADF p-valor:                {est['adf_p_valor']:.4f}")
        print(f"  ADF conclusion:             {est['adf_conclusion']}")
        print(f"  KPSS estadistico:           {est['kpss_estadistico']:.4f}")
        print(f"  KPSS p-valor:               {est['kpss_p_valor']:.4f}")
        print(f"  KPSS conclusion:            {est['kpss_conclusion']}")

    nh = resumen.get("poisson_no_homogeneo")
    if nh and nh.get("aic_homogeneo") is not None:
        print(f"\n  --- Poisson Homogeneo vs No Homogeneo ---")
        print(f"  lambda global:              {nh['lambda_global']:.2f}")
        print(f"  lambda local (casa):        {nh['lambda_local']:.2f}")
        print(f"  lambda visitante (fuera):   {nh['lambda_visitante']:.2f}")
        print(f"  Partidos local:             {nh['n_local']}")
        print(f"  Partidos visitante:         {nh['n_visitante']}")
        print(f"  AIC homogeneo:              {nh['aic_homogeneo']:.2f}")
        print(f"  AIC no homogeneo:           {nh['aic_no_homogeneo']:.2f}")
        print(f"  Conclusion:                 {nh['conclusion']}")

    print(f"\n{'='*60}")
    print(f"  CONCLUSION FINAL")
    print(f"{'='*60}")

    if pp['diferencia_relativa'] < 0.10:
        print("  La propiedad E[X] = Var(X) se cumple adecuadamente.")
        print("  Los datos pueden modelarse mediante un proceso de Poisson.")
        print("  El modelo Poisson es una aproximacion razonable para")
        print("  los puntos anotados por partido de este equipo.")
    elif pp['diferencia_relativa'] < 0.25:
        print("  La propiedad E[X] = Var(X) muestra una desviacion leve.")
        print("  El modelo Poisson es aceptable, aunque se recomienda")
        print("  revisar las graficas CDF y Q-Q para confirmar el ajuste.")
    else:
        print("  La propiedad E[X] = Var(X) presenta desviacion.")
        print("  Se recomienda explorar modelos alternativos como")
        print("  la distribucion binomial negativa (sobredispersion).")

    print(f"{'='*60}\n")


def analyze(df, equipo="Warriors", plots_dir=None, seed=42):
    """
    Funcion principal del modulo de analisis.
    Ejecuta el pipeline completo de analisis Poisson, exponencial,
    estacionariedad y Poisson no homogeneo.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame limpio con columna 'puntos'.
    equipo : str
        Nombre del equipo para titulos de graficas.
    plots_dir : str, optional
        Directorio donde guardar las graficas.
    seed : int
        Semilla para reproducibilidad (reservado).

    Returns
    -------
    tuple
        (resumen dict, stats_dict, propiedad dict, estacionariedad dict,
         no_homogeneo dict, exp_result dict)
    """
    print(f"\n{'='*60}")
    print(f"  ANALISIS DE POISSON — {equipo}")
    print(f"{'='*60}")

    pts_col = "puntos" if "puntos" in df.columns else "PTS"
    puntos = df[pts_col].values.astype(float)

    # 1. Estadistica descriptiva
    print("\n  [1/6] Estadistica descriptiva...")
    stats_dict = estadistica_descriptiva(puntos)
    lambda_est = stats_dict["lambda_estimado"]
    n = stats_dict["n"]

    # 2. Verificar propiedad
    print("  [2/6] Verificando E[X] = Var(X)...")
    propiedad = verificar_propiedad_poisson(stats_dict)

    # 3. Estacionariedad formal
    print("  [3/6] Pruebas de estacionariedad (ADF, KPSS)...")
    estacionariedad = prueba_estacionariedad(puntos)

    # 4. Poisson no homogeneo
    print("  [4/6] Comparando Poisson homogeneo vs no homogeneo...")
    no_homogeneo = analizar_poisson_no_homogeneo(df)

    # 5. Tiempos entre eventos (exponencial)
    print("  [5/6] Identificando tiempos entre eventos...")
    tiempos, umbral, n_eventos = identificar_eventos_y_tiempos(puntos, lambda_est)
    exp_result = {"tiempos": tiempos, "umbral": umbral, "n_eventos": n_eventos}
    if len(tiempos) >= 2:
        lambda_exp = 1.0 / np.mean(tiempos)
        exp_result["lambda_exp"] = lambda_exp
        exp_result["media_tiempos"] = np.mean(tiempos)
    else:
        exp_result["lambda_exp"] = None
        exp_result["media_tiempos"] = None

    # 6. Graficas
    print("  [6/6] Generando graficas...")
    if plots_dir is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        plots_dir = os.path.join(base, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    graficar_histograma_comparativo(puntos, lambda_est, None,
                                     equipo=equipo, plots_dir=plots_dir)
    graficar_qq_poisson(puntos, lambda_est, equipo=equipo, plots_dir=plots_dir)
    graficar_cdf_escalonada(puntos, lambda_est, equipo=equipo,
                             plots_dir=plots_dir)

    fecha_col = "fecha" if "fecha" in df.columns else "GAME_DATE"
    if fecha_col in df.columns:
        try:
            graficar_media_varianza_por_temporada(df, equipo=equipo,
                                                   plots_dir=plots_dir)
        except Exception as e:
            print(f"  (Media/varianza por temporada omitida: {e})")

    if len(tiempos) >= 5 and exp_result["lambda_exp"] is not None:
        graficar_qq_exponencial(tiempos, exp_result["lambda_exp"], umbral,
                                equipo=equipo, plots_dir=plots_dir)
    elif len(tiempos) > 0:
        print(
            "  Pocos eventos para Q-Q exponencial. "
            "Se omite."
        )

    resumen = resumen_estadistico(stats_dict, propiedad,
                                  estacionariedad=estacionariedad,
                                  no_homogeneo=no_homogeneo)
    imprimir_resumen(resumen)

    return resumen, stats_dict, propiedad, estacionariedad, no_homogeneo, exp_result


if __name__ == "__main__":
    # Cargar datos procesados de prueba si existen
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_file = os.path.join(base, "data", "processed", "warriors_game_time_series.csv")
    if os.path.exists(processed_file):
        df_test = pd.read_csv(processed_file)
        analyze(df_test)
    else:
        print("Ejecute primero los módulos extract.py y clean.py")
