"""
PROYECTO: Modelado del número de puntos anotados en partidos de la NBA
         como un proceso de Poisson
=====================================================================

Pipeline principal que orquesta las tres fases del proyecto:
  1. Extracción de datos desde nba_api.
  2. Limpieza y preparación de datos.
  3. Análisis estadístico completo (Poisson).

Uso:
    python main.py                    # Ejecuta con valores por defecto
    python main.py --team lakers     # Analiza a Los Angeles Lakers
    python main.py --help            # Muestra ayuda

Autor: Proyecto Procesos Estocásticos - NBA
"""

import os
import sys
import argparse


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Modelado Poisson de puntos anotados en la NBA"
    )
    parser.add_argument(
        "--team", type=str, default="warriors",
        choices=["warriors", "lakers", "celtics", "bulls", "heat", "spurs",
                 "nuggets", "bucks"],
        help="Equipo a analizar (default: warriors)"
    )
    parser.add_argument(
        "--start-season", type=int, default=2014,
        help="Año de inicio de la primera temporada (default: 2014)"
    )
    parser.add_argument(
        "--end-season", type=int, default=2024,
        help="Año de inicio de la última temporada (default: 2024)"
    )
    parser.add_argument(
        "--skip-extract", action="store_true",
        help="Omite la extracción y usa datos ya guardados"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Semilla para reproducibilidad (default: 42)"
    )
    return parser.parse_args()


def build_seasons(start, end):
    return [f"{y}-{str(y+1)[-2:]}" for y in range(start, end + 1)]


def main():
    args = parse_args()

    from src.extract import extract, get_seasons_list
    from src.clean import clean
    from src.analysis import analyze

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
    seasons = build_seasons(args.start_season, args.end_season)
    seed = args.seed

    print(f"\n{'#'*60}")
    print(f"  PROYECTO: PROCESO DE POISSON EN LA NBA")
    print(f"  Equipo: {equipo_nombre}")
    print(f"  Temporadas: {seasons[0]} a {seasons[-1]} ({len(seasons)} temporadas)")
    print(f"  Semilla: {seed}")
    print(f"{'#'*60}")

    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    plots_dir = os.path.join(PROJECT_ROOT, "plots")

    # --- Fase 1: Extracción ---
    if args.skip_extract:
        print("\n  [FASE 1] Omitiendo extracción (--skip-extract).")
        print(f"  Usando datos en: {raw_dir}")
    else:
        print("\n  [FASE 1] Extrayendo datos de NBA.com...")
        try:
            df_raw, raw_path = extract(
                team_name=team_name,
                seasons=seasons,
                output_dir=raw_dir,
            )
            print(f"  Datos guardados en: {raw_path}")
        except Exception as e:
            print(f"\n  ERROR en extracción: {e}")
            if not os.path.exists(os.path.join(
                    raw_dir, f"{team_name}_game_logs_raw.csv")):
                sys.exit(1)
            print("  Se encontró un archivo previo, continuando con limpieza...")

    # --- Fase 2: Limpieza ---
    print("\n  [FASE 2] Limpiando y preparando datos...")
    try:
        df_clean, clean_path = clean(
            team_name=team_name,
            project_root=PROJECT_ROOT,
        )
    except Exception as e:
        print(f"\n  ERROR en limpieza: {e}")
        sys.exit(1)

    # --- Fase 3: Análisis ---
    print("\n  [FASE 3] Análisis estadístico (Poisson)...")
    resumen, stats_dict, propiedad, chi2_result, ks_result = analyze(
        df_clean,
        equipo=equipo_nombre,
        plots_dir=plots_dir,
        seed=seed,
    )

    # --- Guardar resumen en CSV ---
    summary_path = os.path.join(PROJECT_ROOT, "data", "processed",
                                f"{team_name}_resumen_estadistico.csv")
    import pandas as pd
    summary_rows = [
        {"métrica": "n", "valor": stats_dict["n"]},
        {"métrica": "media_lambda", "valor": stats_dict["media"]},
        {"métrica": "varianza", "valor": stats_dict["varianza"]},
        {"métrica": "desviacion_estandar", "valor": stats_dict["desviacion_estandar"]},
        {"métrica": "minimo", "valor": stats_dict["minimo"]},
        {"métrica": "maximo", "valor": stats_dict["maximo"]},
        {"métrica": "asimetria", "valor": stats_dict["asimetria"]},
        {"métrica": "curtosis", "valor": stats_dict["curtosis"]},
        {"métrica": "ratio_varianza_media", "valor": propiedad["ratio_varianza_media"]},
        {"métrica": "chi2_estadistico", "valor": chi2_result["estadistico_chi2"]},
        {"métrica": "chi2_p_valor", "valor": chi2_result["p_valor"]},
        {"métrica": "chi2_gl", "valor": chi2_result["grados_libertad"]},
        {"métrica": "ks_estadistico", "valor": ks_result["estadistico_ks"]},
        {"métrica": "ks_p_valor_mc", "valor": ks_result["p_valor_monte_carlo"]},
    ]
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\n  Resumen numérico guardado en: {summary_path}")

    print(f"\n{'#'*60}")
    print(f"  PROYECTO COMPLETADO EXITOSAMENTE")
    print(f"  Gráficas guardadas en: {plots_dir}")
    print(f"  Datos procesados en: {processed_dir}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
