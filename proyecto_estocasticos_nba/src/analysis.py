"""
Módulo de Análisis Estadístico — Proceso de Poisson
====================================================
Realiza el análisis completo para modelar los puntos anotados por un equipo
de la NBA como un proceso de Poisson. Incluye:

- Estadística descriptiva (media, varianza, lambda estimado)
- Verificación de la propiedad E[X] = Var(X) = lambda
- Ajuste de la distribución de Poisson (PMF teórica)
- Prueba de bondad de ajuste Chi-cuadrada
- Prueba de Kolmogorov-Smirnov
- Simulación Monte Carlo con numpy.random.poisson
- Comparación visual: histogramas, Q-Q plot, serie temporal, barras chi2

Autor: Proyecto Procesos Estocásticos - NBA
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats
from scipy.stats import poisson


plt.rcParams.update({
    "figure.figsize": (12, 6),
    "figure.dpi": 120,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def estadistica_descriptiva(puntos):
    """
    Calcula estadísticas descriptivas de los puntos anotados.

    Parameters
    ----------
    puntos : array-like
        Serie de puntos anotados por partido.

    Returns
    -------
    dict
        Diccionario con n, media, varianza, desviación estándar,
        lambda estimado, mínimo, máximo, mediana, moda, asimetría y curtosis.
    """
    p = np.array(puntos, dtype=float)
    n = len(p)
    media = np.mean(p)
    varianza = np.var(p, ddof=0)
    desv = np.std(p, ddof=0)

    moda_vals = stats.mode(p, keepdims=False)
    moda = moda_vals.mode if hasattr(moda_vals, 'mode') else moda_vals[0]

    return {
        "n": n,
        "media": media,
        "varianza": varianza,
        "desviacion_estandar": desv,
        "lambda_estimado": media,
        "minimo": np.min(p),
        "maximo": np.max(p),
        "mediana": np.median(p),
        "moda": float(moda),
        "asimetria": stats.skew(p),
        "curtosis": stats.kurtosis(p),
        "ratio_varianza_media": varianza / media if media > 0 else np.nan,
    }


def verificar_propiedad_poisson(stats_dict):
    """
    Verifica si la propiedad E[X] = Var(X) se cumple aproximadamente.
    En una distribución de Poisson, la razón Var(X)/E[X] debe ser cercana a 1.

    Parameters
    ----------
    stats_dict : dict
        Diccionario retornado por estadistica_descriptiva().

    Returns
    -------
    dict
        Diccionario con el diagnóstico de la propiedad.
    """
    media = stats_dict["media"]
    varianza = stats_dict["varianza"]
    ratio = stats_dict["ratio_varianza_media"]
    diferencia_relativa = abs(ratio - 1)

    if diferencia_relativa < 0.10:
        diagnostico = "BUENO: Var(X)/E[X] muy cercano a 1. Propiedad Poisson se cumple."
    elif diferencia_relativa < 0.25:
        diagnostico = "ACEPTABLE: Hay cierta desviación. Posible sobredispersión leve."
    else:
        diagnostico = "DESVIACIÓN SIGNIFICATIVA: Los datos muestran sobredispersión. Poisson puede no ser adecuado."

    return {
        "media": media,
        "varianza": varianza,
        "ratio_varianza_media": ratio,
        "diferencia_relativa": diferencia_relativa,
        "diagnostico": diagnostico,
    }


def poisson_pmf_teorica(lambda_est, x_min=0, x_max=None):
    """
    Calcula la función de masa de probabilidad teórica de Poisson.

    Parameters
    ----------
    lambda_est : float
        Parámetro lambda estimado (media muestral).
    x_min : int
        Valor mínimo del soporte.
    x_max : int, optional
        Valor máximo. Si None, se calcula como lambda + 5*sqrt(lambda).

    Returns
    -------
    tuple
        (array de valores x, array de probabilidades P(X=x))
    """
    if x_max is None:
        x_max = int(lambda_est + 6 * np.sqrt(lambda_est)) + 1

    x_vals = np.arange(x_min, x_max + 1)
    probs = poisson.pmf(x_vals, mu=lambda_est)
    return x_vals, probs


def frecuencias_por_bins(puntos, bins=None):
    """
    Agrupa los puntos en bins y calcula frecuencias observadas.

    Parameters
    ----------
    puntos : array-like
        Serie de puntos anotados por partido.
    bins : array-like, optional
        Bordes de los bins. Si es None, se calculan automáticamente.

    Returns
    -------
    tuple
        (frecuencias observadas, bordes de bins, centros de bins)
    """
    if bins is None:
        p = np.array(puntos)
        min_val = int(np.floor(p.min()))
        max_val = int(np.ceil(p.max()))
        bins = np.arange(min_val - 0.5, max_val + 1.5, 1)

    freqs, edges = np.histogram(puntos, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    return freqs, edges, centers


def prueba_chi_cuadrada(puntos, lambda_est, alpha=0.05):
    """
    Realiza la prueba de bondad de ajuste Chi-cuadrada para la distribución
    de Poisson.

    Agrupa los datos en bins y compara frecuencias observadas vs esperadas.
    Combina bins con frecuencia esperada < 5 para cumplir supuestos.

    Parameters
    ----------
    puntos : array-like
        Serie de puntos anotados por partido.
    lambda_est : float
        Parámetro lambda estimado.
    alpha : float
        Nivel de significancia (default: 0.05).

    Returns
    -------
    dict
        Diccionario con estadístico chi2, p-valor, grados de libertad,
        bins usados, frecuencias observadas/esperadas, y conclusión.
    """
    p = np.array(puntos)
    n = len(p)

    min_val = int(np.floor(p.min()))
    max_val_raw = int(np.ceil(p.max()))
    tail_val = int(lambda_est + 4 * np.sqrt(lambda_est)) + 1
    max_val = max(max_val_raw, tail_val)

    observed = []
    expected = []
    bin_labels = []
    current_obs = 0
    current_exp = 0
    bin_start = min_val

    for k in range(min_val, max_val + 1):
        obs_k = np.sum(p == k)
        exp_k = n * poisson.pmf(k, mu=lambda_est)
        current_obs += obs_k
        current_exp += exp_k

        if current_exp >= 5 or k == max_val:
            observed.append(current_obs)
            expected.append(current_exp)
            if bin_start == k:
                bin_labels.append(str(k))
            else:
                bin_labels.append(f"{bin_start}-{k}")
            current_obs = 0
            current_exp = 0
            bin_start = k + 1

    observed = np.array(observed)
    expected = np.array(expected)
    expected = expected * np.sum(observed) / np.sum(expected)

    if np.any(expected == 0):
        mask = expected > 0
        observed = observed[mask]
        expected = expected[mask]

    if len(observed) < 3:
        return {
            "estadistico_chi2": np.nan,
            "p_valor": np.nan,
            "grados_libertad": np.nan,
            "bins_labels": bin_labels,
            "observados": observed.tolist(),
            "esperados": expected.tolist(),
            "conclusion": "Muy pocos bins para realizar la prueba chi-cuadrada.",
            "nivel_significancia": alpha,
            "rechazar_h0": None,
        }

    chi2_stat, p_valor = stats.chisquare(f_obs=observed, f_exp=expected)
    df_gl = len(observed) - 2

    chi2_critico = stats.chi2.ppf(1 - alpha, df_gl)
    rechazar = p_valor < alpha

    if rechazar:
        conclusion = (
            f"Se RECHAZA H0 (p={p_valor:.4f} < alpha={alpha}). "
            f"Los datos NO siguen una distribución de Poisson."
        )
    else:
        conclusion = (
            f"NO se rechaza H0 (p={p_valor:.4f} >= alpha={alpha}). "
            f"Los datos son consistentes con una distribución de Poisson."
        )

    return {
        "estadistico_chi2": chi2_stat,
        "p_valor": p_valor,
        "chi2_critico": chi2_critico,
        "grados_libertad": df_gl,
        "bins_labels": bin_labels,
        "observados": observed.tolist(),
        "esperados": expected.tolist(),
        "conclusion": conclusion,
        "nivel_significancia": alpha,
        "rechazar_h0": rechazar,
    }


def prueba_ks(puntos, lambda_est, alpha=0.05, n_sim=10000):
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


def graficar_histograma_comparativo(puntos_reales, lambda_est, puntos_simulados,
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
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Panel izquierdo: Reales + PMF teórica ---
    ax = axes[0]
    p_real = np.array(puntos_reales).astype(int)
    min_val = min(p_real.min(), int(lambda_est - 4 * np.sqrt(lambda_est)))
    max_val = max(p_real.max(), int(lambda_est + 4 * np.sqrt(lambda_est)))
    bins = np.arange(min_val - 0.5, max_val + 1.5, 1)

    ax.hist(p_real, bins=bins, density=True, alpha=0.65, color="#2196F3",
            edgecolor="white", linewidth=0.5, label="Datos reales")

    x_pmf, y_pmf = poisson_pmf_teorica(lambda_est, x_min=min_val, x_max=max_val)
    ax.plot(x_pmf, y_pmf, "o-", color="#D32F2F", linewidth=2, markersize=4,
            label=f"Poisson teórica (lambda={lambda_est:.1f})")

    ax.axvline(lambda_est, color="#D32F2F", linestyle="--", alpha=0.5,
               label=f"Media = {lambda_est:.1f}")
    ax.set_xlabel("Puntos por partido")
    ax.set_ylabel("Densidad de probabilidad")
    ax.set_title(f"Datos Reales vs Poisson Teórica — {equipo}")
    ax.legend(loc="upper right")
    ax.yaxis.set_major_locator(MaxNLocator(integer=False))

    # --- Panel derecho: Reales vs Simulados ---
    ax = axes[1]
    p_sim = np.array(puntos_simulados).astype(int)
    both = np.concatenate([p_real, p_sim])
    min_all = both.min()
    max_all = both.max()
    bins_all = np.arange(min_all - 0.5, max_all + 1.5, 1)

    ax.hist(p_real, bins=bins_all, density=True, alpha=0.55, color="#2196F3",
            edgecolor="white", linewidth=0.5, label="Datos reales")
    ax.hist(p_sim, bins=bins_all, density=True, alpha=0.45, color="#FF9800",
            edgecolor="white", linewidth=0.5, label="Simulación Poisson")

    ax.set_xlabel("Puntos por partido")
    ax.set_ylabel("Densidad de probabilidad")
    ax.set_title(f"Reales vs Simulados (Poisson, lambda={lambda_est:.1f}) — {equipo}")
    ax.legend(loc="upper right")

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
    ax.set_title(f"Q-Q Plot Poisson — {equipo} (lambda={lambda_est:.1f})")
    ax.legend(loc="upper left")
    ax.set_aspect("equal")

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
    ax.set_title(f"Media y Varianza de Puntos por Temporada — {equipo}")
    ax.legend(loc="upper left")

    plt.tight_layout()

    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
    filepath = os.path.join(plots_dir, "media_varianza_por_temporada.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gráfica guardada: {filepath}")
    return filepath


def resumen_estadistico(stats_dict, propiedad, chi2_result, ks_result):
    """
    Genera un resumen de todos los resultados estadísticos en formato
    de diccionario, listo para mostrar o exportar.

    Parameters
    ----------
    stats_dict : dict
        Resultado de estadistica_descriptiva().
    propiedad : dict
        Resultado de verificar_propiedad_poisson().
    chi2_result : dict
        Resultado de prueba_chi_cuadrada().
    ks_result : dict
        Resultado de prueba_ks().

    Returns
    -------
    dict
        Resumen completo del análisis.
    """
    return {
        "estadistica_descriptiva": stats_dict,
        "propiedad_poisson": propiedad,
        "prueba_chi_cuadrada": chi2_result,
        "prueba_kolmogorov_smirnov": ks_result,
    }


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
    chi2 = resumen["prueba_chi_cuadrada"]
    ks = resumen["prueba_kolmogorov_smirnov"]

    print(f"\n{'='*60}")
    print(f"  RESUMEN DEL ANÁLISIS ESTADÍSTICO")
    print(f"{'='*60}")

    print(f"\n  --- Estadística Descriptiva ---")
    print(f"  Partidos analizados (n):    {sd['n']}")
    print(f"  Media muestral (lambda):    {sd['media']:.2f}")
    print(f"  Varianza muestral:          {sd['varianza']:.2f}")
    print(f"  Desviación estándar:        {sd['desviacion_estandar']:.2f}")
    print(f"  Mínimo:                     {sd['minimo']:.0f}")
    print(f"  Máximo:                     {sd['maximo']:.0f}")
    print(f"  Asimetría:                  {sd['asimetria']:.3f}")
    print(f"  Curtosis:                   {sd['curtosis']:.3f}")

    print(f"\n  --- Propiedad E[X] = Var(X) = lambda ---")
    print(f"  Razón Var(X)/E[X]:          {pp['ratio_varianza_media']:.4f}")
    print(f"  Diferencia relativa:        {pp['diferencia_relativa']:.4f}")
    print(f"  Diagnóstico:                {pp['diagnostico']}")

    print(f"\n  --- Prueba Chi-Cuadrada (alpha = {chi2['nivel_significancia']}) ---")
    print(f"  Estadístico chi2:           {chi2['estadistico_chi2']:.3f}")
    print(f"  Grados de libertad:         {chi2['grados_libertad']}")
    print(f"  P-valor:                    {chi2['p_valor']:.4f}")
    print(f"  Conclusión:                 {chi2['conclusion']}")

    print(f"\n  --- Prueba Kolmogorov-Smirnov (alpha = {ks['nivel_significancia']}) ---")
    print(f"  Estadístico KS:             {ks['estadistico_ks']:.4f}")
    print(f"  P-valor Monte Carlo:        {ks['p_valor_monte_carlo']:.4f}")
    print(f"  Simulaciones MC:            {ks['n_simulaciones']}")
    print(f"  Conclusión:                 {ks['conclusion']}")

    print(f"\n{'='*60}")
    print(f"  CONCLUSIÓN FINAL")
    print(f"{'='*60}")

    rechazos = 0
    if chi2.get("rechazar_h0"):
        rechazos += 1
    if ks.get("rechazar_h0"):
        rechazos += 1

    if rechazos == 0:
        print("  Ambas pruebas estadísticas indican que los datos pueden")
        print("  modelarse adecuadamente mediante un proceso de Poisson.")
        print("  El modelo Poisson es una aproximación razonable para")
        print("  los puntos anotados por partido de este equipo.")
    elif rechazos == 1:
        print("  Los resultados son mixtos. Una prueba rechaza H0 y la")
        print("  otra no. Se recomienda explorar modelos alternativos como")
        print("  la distribución binomial negativa (para sobredispersión).")
    else:
        print("  Ambas pruebas rechazan la hipótesis nula. Los datos NO")
        print("  siguen una distribución de Poisson. Se recomienda explorar")
        print("  modelos alternativos (binomial negativa, Poisson compuesto).")

    print(f"{'='*60}\n")


def analyze(df, equipo="Warriors", plots_dir=None, seed=42):
    """
    Función principal del módulo de análisis.
    Ejecuta el pipeline completo de análisis Poisson.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame limpio con columna 'puntos'.
    equipo : str
        Nombre del equipo para títulos de gráficas.
    plots_dir : str, optional
        Directorio donde guardar las gráficas.
    seed : int
        Semilla para reproducibilidad de simulaciones.

    Returns
    -------
    tuple
        (resumen dict, stats_dict, propiedad dict, chi2 dict, ks dict)
    """
    print(f"\n{'='*60}")
    print(f"  ANÁLISIS DE POISSON — {equipo}")
    print(f"{'='*60}")

    pts_col = "puntos" if "puntos" in df.columns else "PTS"
    puntos = df[pts_col].values.astype(float)

    # 1. Estadística descriptiva
    print("\n  [1/7] Estadística descriptiva...")
    stats_dict = estadistica_descriptiva(puntos)
    lambda_est = stats_dict["lambda_estimado"]
    n = stats_dict["n"]

    # 2. Verificar propiedad
    print("  [2/7] Verificando E[X] = Var(X)...")
    propiedad = verificar_propiedad_poisson(stats_dict)

    # 3. Ajuste Poisson (PMF teórica)
    print("  [3/7] Ajustando distribución de Poisson...")
    x_pmf, y_pmf = poisson_pmf_teorica(lambda_est)

    # 4. Prueba Chi-cuadrada
    print("  [4/7] Prueba de bondad de ajuste Chi-cuadrada...")
    chi2_result = prueba_chi_cuadrada(puntos, lambda_est)

    # 5. Prueba KS
    print("  [5/7] Prueba de Kolmogorov-Smirnov (Monte Carlo)...")
    ks_result = prueba_ks(puntos, lambda_est)

    # 6. Simulación
    print("  [6/7] Simulando datos con Poisson...")
    puntos_sim = simular_poisson(lambda_est, n, seed=seed)

    # 7. Gráficas
    print("  [7/7] Generando gráficas...")
    if plots_dir is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        plots_dir = os.path.join(base, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    graficar_histograma_comparativo(puntos, lambda_est, puntos_sim,
                                     equipo=equipo, plots_dir=plots_dir)
    graficar_qq_poisson(puntos, lambda_est, equipo=equipo, plots_dir=plots_dir)
    graficar_serie_temporal(df, equipo=equipo, plots_dir=plots_dir)
    graficar_barras_chi2(chi2_result, lambda_est, n, equipo=equipo,
                          plots_dir=plots_dir)

    fecha_col = "fecha" if "fecha" in df.columns else "GAME_DATE"
    if fecha_col in df.columns:
        try:
            graficar_media_varianza_por_temporada(df, equipo=equipo,
                                                   plots_dir=plots_dir)
        except Exception as e:
            print(f"  (Media/varianza por temporada omitida: {e})")

    resumen = resumen_estadistico(stats_dict, propiedad, chi2_result, ks_result)
    imprimir_resumen(resumen)

    return resumen, stats_dict, propiedad, chi2_result, ks_result


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.clean import clean

    df_clean, _ = clean(team_name="warriors")
    analyze(df_clean, equipo="Golden State Warriors")
