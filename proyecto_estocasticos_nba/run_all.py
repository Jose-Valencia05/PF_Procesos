"""
Script: Ejecutar analisis para TODOS los equipos
================================================
Genera una carpeta results/<equipo>/ con:
  - plots/*.png   (5 graficas)
  - resumen.csv   (todas las metricas)
  - markov.csv    (matriz de transicion y prediccion)
  - resumen.ipynb (notebook con resultados del equipo)

Al final genera:
  - results/comparacion_equipos.csv     (tabla comparativa)
  - results/comparacion_equipos.ipynb   (notebook maestro)
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.extract import extract
from src.clean import clean
from src.analysis import analyze
from src.markov import markov

TEAMS = {
    "warriors": "Golden State Warriors",
    "lakers": "Los Angeles Lakers",
    "celtics": "Boston Celtics",
    "bulls": "Chicago Bulls",
    "heat": "Miami Heat",
    "spurs": "San Antonio Spurs",
    "nuggets": "Denver Nuggets",
    "bucks": "Milwaukee Bucks",
}

SEASONS = ["2023-24", "2024-25"]
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def crear_notebook_equipo(nombre_corto, nombre_largo, resultados, plots_rel):
    """Crea un notebook ipynb sencillo con los resultados de un equipo."""
    cells = []

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# Resultados: {nombre_largo} ({nombre_corto})\n",
            "\n",
            f"Temporadas: {SEASONS[0]} a {SEASONS[-1]} (2 temporadas)\n",
            "\n",
            "---\n",
            "## Resumen Estadistico\n",
        ]
    })

    stats_rows = ""
    for m in [
        ("n", "Partidos analizados"),
        ("media_lambda", "Media (lambda)"),
        ("varianza", "Varianza"),
        ("desviacion_estandar", "Desviacion estandar"),
        ("minimo", "Minimo"),
        ("maximo", "Maximo"),
        ("asimetria", "Asimetria"),
        ("curtosis", "Curtosis"),
        ("ratio_varianza_media", "Razon Var/Media"),
    ]:
        stats_rows += f"| {m[1]} | {resultados[m[0]]} |\n"

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "| Metrica | Valor |\n",
            "|---------|-------|\n",
            stats_rows,
        ]
    })

    if resultados.get("adf_p_valor") is not None:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Estacionariedad\n",
                "\n",
                f"| Prueba | Estadistico | P-valor | Conclusion |\n",
                f"| ADF | {resultados['adf_estadistico']:.3f} | {resultados['adf_p_valor']:.4f} | {resultados['adf_conclusion']} |\n",
                f"| KPSS | {resultados['kpss_estadistico']:.3f} | {resultados['kpss_p_valor']:.4f} | {resultados['kpss_conclusion']} |\n",
            ]
        })

    if resultados.get("aic_homogeneo") is not None:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Poisson No Homogeneo\n",
                "\n",
                f"| Metrica | Valor |\n",
                f"| lambda global | {resultados['lambda_global']:.2f} |\n",
                f"| lambda local | {resultados['lambda_local']:.2f} |\n",
                f"| lambda visitante | {resultados['lambda_visitante']:.2f} |\n",
                f"| AIC homogeneo | {resultados['aic_homogeneo']:.2f} |\n",
                f"| AIC no homogeneo | {resultados['aic_no_homogeneo']:.2f} |\n",
                f"| Conclusion | {resultados['conclusion_no_homogeneo']} |\n",
            ]
        })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Cadenas de Markov\n",
            "\n",
            "| Metrica | Valor |\n",
            "| Matriz de transicion P | |\n",
            f"| P(W\\|W) | {resultados['P_WW']:.3f} |\n",
            f"| P(L\\|W) | {resultados['P_WL']:.3f} |\n",
            f"| P(W\\|L) | {resultados['P_LW']:.3f} |\n",
            f"| P(L\\|L) | {resultados['P_LL']:.3f} |\n",
            f"| pi_W estacionaria | {resultados['pi_W']:.3f} |\n",
            f"| pi_L estacionaria | {resultados['pi_L']:.3f} |\n",
            f"| Proporcion W observada | {resultados['prop_W']:.3f} |\n",
            f"| Ultimo resultado | {resultados['ultimo']} |\n",
            f"| P(siguiente = W) | {resultados['prob_W_sig']:.3f} |\n",
            f"| Prediccion | {resultados['prediccion']} |\n",
        ]
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Graficas\n",
            "\n",
            "Todas las graficas estan en la carpeta `plots/`.\n",
        ]
    })

    for nombre, desc in [
        ("histograma_comparativo.png", "Histograma real vs PMF teorica de Poisson"),
        ("cdf_poisson.png", "CDF empirica vs CDF teorica (escalones)"),
        ("qq_plot_poisson.png", "Q-Q Plot Poisson"),
        ("media_varianza_por_temporada.png", "Estabilidad de lambda por temporada"),
        ("qq_exponencial.png", "Q-Q Plot Exponencial (tiempos entre eventos)"),
        ("markov_transiciones.png", "Diagrama de transicion de Markov"),
    ]:
        path = os.path.join(plots_rel, nombre)
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"### {desc}\n",
                f"![{desc}]({path})\n",
            ]
        })

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    return nb


def procesar_equipo(nombre_corto, nombre_largo):
    """Ejecuta el pipeline completo para un equipo."""
    t_start = time.time()

    out_dir = os.path.join(RESULTS_DIR, nombre_corto)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  EQUIPO: {nombre_largo} ({nombre_corto})")
    print(f"{'#'*60}")

    raw_path = os.path.join(PROJECT_ROOT, "data", "raw",
                            f"{nombre_corto}_game_logs_raw.csv")

    if not os.path.exists(raw_path):
        print(f"  Extrayendo datos para {nombre_largo}...")
        try:
            extract(team_name=nombre_corto, seasons=SEASONS,
                    output_dir=os.path.join(PROJECT_ROOT, "data", "raw"))
        except Exception as e:
            print(f"  ERROR en extraccion: {e}")
            return None

    try:
        df_clean, _ = clean(
            team_name=nombre_corto,
            project_root=PROJECT_ROOT,
            seasons=SEASONS,
        )
    except Exception as e:
        print(f"  ERROR en limpieza: {e}")
        return None

    resumen, stats_dict, propiedad, estacionariedad, no_homogeneo, exp_result = analyze(
        df_clean, equipo=nombre_largo, plots_dir=plots_dir
    )

    markov_result = markov(df_clean, equipo=nombre_largo, plots_dir=plots_dir)

    elapsed = time.time() - t_start
    print(f"\n  Equipo {nombre_corto} completado en {elapsed:.1f}s")

    resultados = {
        "equipo": nombre_corto,
        "nombre": nombre_largo,
        "n": stats_dict["n"],
        "media_lambda": stats_dict["media"],
        "varianza": stats_dict["varianza"],
        "desviacion_estandar": stats_dict["desviacion_estandar"],
        "minimo": stats_dict["minimo"],
        "maximo": stats_dict["maximo"],
        "asimetria": stats_dict["asimetria"],
        "curtosis": stats_dict["curtosis"],
        "ratio_varianza_media": propiedad["ratio_varianza_media"],
    }

    if estacionariedad.get("adf_estadistico") is not None:
        resultados.update({
            "adf_estadistico": estacionariedad["adf_estadistico"],
            "adf_p_valor": estacionariedad["adf_p_valor"],
            "adf_conclusion": estacionariedad["adf_conclusion"],
            "kpss_estadistico": estacionariedad["kpss_estadistico"],
            "kpss_p_valor": estacionariedad["kpss_p_valor"],
            "kpss_conclusion": estacionariedad["kpss_conclusion"],
        })

    if no_homogeneo.get("aic_homogeneo") is not None:
        resultados.update({
            "lambda_global": no_homogeneo["lambda_global"],
            "lambda_local": no_homogeneo["lambda_local"],
            "lambda_visitante": no_homogeneo["lambda_visitante"],
            "aic_homogeneo": no_homogeneo["aic_homogeneo"],
            "aic_no_homogeneo": no_homogeneo["aic_no_homogeneo"],
            "conclusion_no_homogeneo": no_homogeneo["conclusion"],
        })

    if exp_result.get("lambda_exp") is not None:
        resultados.update({
            "umbral_evento": exp_result["umbral"],
            "n_eventos": exp_result["n_eventos"],
            "lambda_exp": exp_result["lambda_exp"],
            "media_tiempos_entre_eventos": exp_result["media_tiempos"],
        })

    resultados.update({
        "P_WW": markov_result["matriz_transicion"][0][0],
        "P_WL": markov_result["matriz_transicion"][0][1],
        "P_LW": markov_result["matriz_transicion"][1][0],
        "P_LL": markov_result["matriz_transicion"][1][1],
        "pi_W": markov_result["pi_W"],
        "pi_L": markov_result["pi_L"],
        "prop_W": markov_result["proporcion_W_observada"],
        "ultimo": markov_result["ultimo_resultado"],
        "prob_W_sig": markov_result["probabilidad_W_siguiente"],
        "prediccion": markov_result["prediccion"],
    })

    pd.DataFrame([{k: v for k, v in resultados.items()
                     if isinstance(v, (int, float, str))}]).to_csv(
        os.path.join(out_dir, "resumen.csv"), index=False)

    pd.DataFrame([
        {"metrica": k, "valor": v}
        for k, v in markov_result.items()
        if isinstance(v, (int, float, str))
    ]).to_csv(os.path.join(out_dir, "markov.csv"), index=False)

    nb = crear_notebook_equipo(nombre_corto, nombre_largo, resultados, "plots")
    nb_path = os.path.join(out_dir, "resumen.ipynb")
    with open(nb_path, "w") as f:
        json.dump(nb, f, indent=2)
    print(f"  Notebook guardado: {nb_path}")

    return resultados


def generar_notebook_comparativo(todos):
    """Genera un notebook maestro comparando todos los equipos."""
    cells = []

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Comparacion de Todos los Equipos\n",
            "\n",
            f"Temporadas: {SEASONS[0]} a {SEASONS[-1]} (2 temporadas)\n",
            "\n",
            "---\n",
            "## 1. Resumen General\n",
        ]
    })

    df = pd.DataFrame([r for r in todos if r is not None])

    headers = ["Equipo", "n", "Media (λ)", "Varianza", "Var/Media", "ADF p", "KPSS p",
               "AIC Hom", "AIC NoHom", "pi_W", "P(W|W)", "P(W|L)", "Prediccion"]
    table = "| " + " | ".join(headers) + " |\n"
    table += "|" + "|".join(["---" for _ in headers]) + "|\n"
    for _, row in df.iterrows():
        vals = [
            row.get("nombre", row["equipo"]),
            str(row["n"]),
            f"{row['media_lambda']:.1f}",
            f"{row['varianza']:.1f}",
            f"{row.get('ratio_varianza_media', 0):.3f}",
            f"{row.get('adf_p_valor', 'N/A'):.4f}" if row.get('adf_p_valor') is not None else "N/A",
            f"{row.get('kpss_p_valor', 'N/A'):.4f}" if row.get('kpss_p_valor') is not None else "N/A",
            f"{row.get('aic_homogeneo', 'N/A'):.1f}" if row.get('aic_homogeneo') is not None else "N/A",
            f"{row.get('aic_no_homogeneo', 'N/A'):.1f}" if row.get('aic_no_homogeneo') is not None else "N/A",
            f"{row.get('pi_W', 0):.3f}",
            f"{row.get('P_WW', 0):.3f}",
            f"{row.get('P_LW', 0):.3f}",
            str(row.get("prediccion", "")),
        ]
        table += "| " + " | ".join(vals) + " |\n"

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [table]
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## 2. Conclusiones\n",
            "\n",
            "Todos los equipos fueron analizados con el mismo pipeline:\n",
            "- Modelo Poisson (homogeneo y no homogeneo local/visitante)\n",
            "- Pruebas de estacionariedad (ADF + KPSS)\n",
            "- Distribucion exponencial de tiempos entre eventos de altos puntos\n",
            "- Cadenas de Markov W/L con matriz de transicion y prediccion\n",
            "\n",
            "Los resultados detallados de cada equipo estan en su carpeta individual\n",
            "dentro de `results/`.\n",
        ]
    })

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    return nb


def main():
    print(f"\n{'#'*60}")
    print(f"  ANALISIS DE TODOS LOS EQUIPOS")
    print(f"  Temporadas: {SEASONS[0]} a {SEASONS[-1]}")
    print(f"  {len(TEAMS)} equipos")
    print(f"{'#'*60}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    todos_resultados = []
    inicio = time.time()

    for i, (corto, largo) in enumerate(TEAMS.items()):
        print(f"\n[{i+1}/{len(TEAMS)}] Procesando {largo}...")
        r = procesar_equipo(corto, largo)
        todos_resultados.append(r)

    df_todos = pd.DataFrame([r for r in todos_resultados if r is not None])
    csv_path = os.path.join(RESULTS_DIR, "comparacion_equipos.csv")
    df_todos.to_csv(csv_path, index=False)

    nb = generar_notebook_comparativo(todos_resultados)
    nb_path = os.path.join(RESULTS_DIR, "comparacion_equipos.ipynb")
    with open(nb_path, "w") as f:
        json.dump(nb, f, indent=2)

    total = time.time() - inicio
    print(f"\n{'#'*60}")
    print(f"  TODOS LOS EQUIPOS COMPLETADOS")
    print(f"  Tiempo total: {total:.1f}s")
    print(f"  Resultados en: {RESULTS_DIR}")
    print(f"  CSV comparativo: {csv_path}")
    print(f"  Notebook comparativo: {nb_path}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
