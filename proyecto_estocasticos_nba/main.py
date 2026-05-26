"""
PROYECTO: Modelado del numero de puntos anotados en partidos de la NBA
         como un proceso de Poisson + Cadenas de Markov
=====================================================================

Pipeline principal que orquesta las fases del proyecto:
  1. Extraccion de datos desde nba_api.
  2. Limpieza y preparacion de datos.
  3. Analisis estadistico completo (Poisson, exponencial, estacionariedad).
  4. Cadenas de Markov (resultados W/L).

Uso:
    python main.py                       # Corre con valores por defecto (Warriors, 5 partidos)
    python main.py --team lakers --limit-games 3
    python main.py --skip-extract        # Usa archivos CSV locales ya descargados

Autor: Proyecto Procesos Estocasticos - NBA
"""

import os
import sys
import argparse
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Modelado Poisson + Markov de la NBA"
    )
    parser.add_argument(
        "--team", type=str, default="warriors",
        choices=["warriors", "lakers", "celtics", "bulls", "heat", "spurs",
                 "nuggets", "bucks"],
        help="Equipo a analizar (default: warriors)"
    )
    parser.add_argument(
        "--start-season", type=int, default=2023,
        help="Anio de inicio de la primera temporada (default: 2023)"
    )
    parser.add_argument(
        "--end-season", type=int, default=2024,
        help="Anio de inicio de la ultima temporada (default: 2024)"
    )
    parser.add_argument(
        "--skip-extract", action="store_true",
        help="Omite la extraccion y usa datos ya guardados"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Semilla para reproducibilidad (default: 42)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Importación dinámica después del path insert
    from src.extract import extract
    from src.clean import clean
    from src.analysis import analyze
    from src.markov import markov

    TEAM_NAMES = {
        "warriors": "Golden State Warriors",
        "lakers": "Los Angeles Lakers",
        "celtics": "Boston Celtics",
        "bulls": "Chicago Bulls",
        "heat": "Miami Heat",
        "spurs": "San Antonio Spurs",
        "nuggets": "Denver Nuggets",
        "bucks": "Milwaukee Bucks",
    }

    team_name = args.team
    equipo_nombre = TEAM_NAMES.get(team_name, team_name.title())

    print(f"\n{'#'*60}")
    print(f"  PROYECTO: PROCESOS ESTOCASTICOS EN LA NBA")
    print(f"  Equipo: {equipo_nombre}")
    print(f"  Temporadas: {seasons[0]} a {seasons[-1]} ({len(seasons)} temporadas)")
    print(f"  Semilla: {seed}")
    print(f"{'#'*60}")

    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    plots_dir = os.path.join(PROJECT_ROOT, "plots")

    # --- Fase 1: Extraccion ---
    if args.skip_extract:
        print("\n  [FASE 1] Omitiendo extraccion (--skip-extract).")
        print(f"  Usando datos en: {raw_dir}")
    else:
        print("\n  [FASE 1] Extrayendo datos Play-by-Play de NBA.com...")
        try:
            df_raw, raw_path = extract(
                team_name=team_name,
                limit_games=args.limit_games,
                output_dir=raw_dir,
            )
            print(f"  Datos Play-by-Play guardados en: {raw_path}")
        except Exception as e:
            print(f"\n  ERROR en extraccion: {e}")
            if not os.path.exists(os.path.join(
                    raw_dir, f"{team_name}_game_logs_raw.csv")):
                sys.exit(1)
            print("  Se encontro un archivo previo, continuando con limpieza...")

    # --- Fase 2: Limpieza y Serie Temporal ---
    print("\n  [FASE 2] Limpiando y convirtiendo a serie temporal por minutos...")
    try:
        df_clean, clean_path = clean(
            team_name=team_name,
            project_root=PROJECT_ROOT,
            seasons=seasons,
        )
    except Exception as e:
        print(f"\n  ERROR en limpieza y procesamiento: {e}")
        sys.exit(1)

    # --- Fase 3: Analisis (Poisson, exponencial, estacionariedad) ---
    print("\n  [FASE 3] Analisis estadistico (Poisson)...")
    (resumen, stats_dict, propiedad,
     estacionariedad, no_homogeneo, exp_result) = analyze(
        df_clean,
        equipo=equipo_nombre,
        plots_dir=plots_dir,
        seed=seed,
    )

    # --- Fase 4: Cadenas de Markov ---
    print("\n  [FASE 4] Cadenas de Markov (W/L)...")
    markov_result = markov(df_clean, equipo=equipo_nombre,
                           plots_dir=plots_dir)

    # --- Guardar resumen en CSV ---
    summary_path = os.path.join(PROJECT_ROOT, "data", "processed",
                                f"{team_name}_resumen_estadistico.csv")
    import pandas as pd
    summary_rows = [
        {"metrica": "n", "valor": stats_dict["n"]},
        {"metrica": "media_lambda", "valor": stats_dict["media"]},
        {"metrica": "varianza", "valor": stats_dict["varianza"]},
        {"metrica": "desviacion_estandar", "valor": stats_dict["desviacion_estandar"]},
        {"metrica": "minimo", "valor": stats_dict["minimo"]},
        {"metrica": "maximo", "valor": stats_dict["maximo"]},
        {"metrica": "asimetria", "valor": stats_dict["asimetria"]},
        {"metrica": "curtosis", "valor": stats_dict["curtosis"]},
        {"metrica": "ratio_varianza_media", "valor": propiedad["ratio_varianza_media"]},
    ]

    if estacionariedad.get("adf_estadistico") is not None:
        summary_rows.extend([
            {"metrica": "adf_estadistico", "valor": estacionariedad["adf_estadistico"]},
            {"metrica": "adf_p_valor", "valor": estacionariedad["adf_p_valor"]},
            {"metrica": "kpss_estadistico", "valor": estacionariedad["kpss_estadistico"]},
            {"metrica": "kpss_p_valor", "valor": estacionariedad["kpss_p_valor"]},
        ])

    if no_homogeneo.get("aic_homogeneo") is not None:
        summary_rows.extend([
            {"metrica": "lambda_local", "valor": no_homogeneo["lambda_local"]},
            {"metrica": "lambda_visitante", "valor": no_homogeneo["lambda_visitante"]},
            {"metrica": "aic_homogeneo", "valor": no_homogeneo["aic_homogeneo"]},
            {"metrica": "aic_no_homogeneo", "valor": no_homogeneo["aic_no_homogeneo"]},
        ])

    if exp_result.get("lambda_exp") is not None:
        summary_rows.extend([
            {"metrica": "umbral_evento", "valor": exp_result["umbral"]},
            {"metrica": "n_eventos", "valor": exp_result["n_eventos"]},
            {"metrica": "lambda_exp", "valor": exp_result["lambda_exp"]},
            {"metrica": "media_tiempos_entre_eventos", "valor": exp_result["media_tiempos"]},
        ])

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\n  Resumen numerico guardado en: {summary_path}")

    # --- Guardar resumen de Markov en CSV ---
    markov_path = os.path.join(PROJECT_ROOT, "data", "processed",
                               f"{team_name}_markov.csv")
    markov_rows = [
        {"metrica": "P_W_dado_W", "valor": markov_result["matriz_transicion"][0][0]},
        {"metrica": "P_L_dado_W", "valor": markov_result["matriz_transicion"][0][1]},
        {"metrica": "P_W_dado_L", "valor": markov_result["matriz_transicion"][1][0]},
        {"metrica": "P_L_dado_L", "valor": markov_result["matriz_transicion"][1][1]},
        {"metrica": "pi_W_estacionaria", "valor": markov_result["pi_W"]},
        {"metrica": "pi_L_estacionaria", "valor": markov_result["pi_L"]},
        {"metrica": "proporcion_W_observada", "valor": markov_result["proporcion_W_observada"]},
        {"metrica": "ultimo_resultado", "valor": markov_result["ultimo_resultado"]},
        {"metrica": "prob_W_siguiente", "valor": markov_result["probabilidad_W_siguiente"]},
        {"metrica": "prediccion_siguiente", "valor": markov_result["prediccion"]},
    ]
    pd.DataFrame(markov_rows).to_csv(markov_path, index=False)
    print(f"  Resumen Markov guardado en: {markov_path}")

    print(f"\n{'#'*60}")
    print(f"  PROYECTO COMPLETADO EXITOSAMENTE")
    print(f"  Graficas guardadas en: {plots_dir}")
    print(f"  Datos procesados en: {processed_dir}")
    print(f"{'#'*60}\n")

if __name__ == "__main__":
    main()
