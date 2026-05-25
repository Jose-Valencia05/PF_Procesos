"""
PROYECTO: Detección Estocástica de Rachas y Momentos de Tiempo Fuera (Poisson)
=============================================================================

Pipeline principal que orquesta las tres fases del proyecto:
  1. Extracción de datos Play-by-Play de la temporada 2024-25 desde nba_api.
  2. Limpieza y preparación de series temporales por minuto.
  3. Análisis estocástico (Poisson) para sugerir Tiempos Fuera (Timeouts).

Uso:
    python main.py                       # Corre con valores por defecto (Warriors, 5 partidos)
    python main.py --team lakers --limit-games 3
    python main.py --skip-extract        # Usa archivos CSV locales ya descargados

Autor: Proyecto Procesos Estocásticos - NBA
"""

import os
import sys
import argparse
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Modelado Poisson de rachas y detección estocástica de Timeouts en la NBA"
    )
    parser.add_argument(
        "--team", type=str, default="warriors",
        choices=["warriors", "lakers", "celtics", "bulls", "heat", "spurs",
                 "nuggets", "bucks"],
        help="Equipo a analizar (default: warriors)"
    )
    parser.add_argument(
        "--limit-games", type=int, default=5,
        help="Cantidad de partidos a procesar en la extracción (default: 5)"
    )
    parser.add_argument(
        "--skip-extract", action="store_true",
        help="Omite la extracción de la API y utiliza datos ya guardados"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Importación dinámica después del path insert
    from src.extract import extract
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

    print(f"\n{'#'*60}")
    print(f"  PROYECTO: PROCESO DE POISSON Y TIMEOUTS EN LA NBA")
    print(f"  Equipo objetivo: {equipo_nombre}")
    print(f"  Temporada: 2024-25")
    print(f"{'#'*60}")

    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    plots_dir = os.path.join(PROJECT_ROOT, "plots")

    # --- Fase 1: Extracción ---
    if args.skip_extract:
        print("\n  [FASE 1] Omitiendo extracción (--skip-extract).")
        print(f"  Buscando datos existentes en: {raw_dir}")
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
            print(f"\n  ERROR en extracción: {e}")
            pbp_path = os.path.join(raw_dir, f"{team_name}_pbp_raw.csv")
            if not os.path.exists(pbp_path):
                print("  No se encontró un archivo previo de PBP. Cancelando ejecución.")
                sys.exit(1)
            print("  Se encontró un archivo previo, continuando con limpieza...")

    # --- Fase 2: Limpieza y Serie Temporal ---
    print("\n  [FASE 2] Limpiando y convirtiendo a serie temporal por minutos...")
    try:
        df_clean, clean_path = clean(
            team_name=team_name,
            project_root=PROJECT_ROOT,
        )
    except Exception as e:
        print(f"\n  ERROR en limpieza y procesamiento: {e}")
        sys.exit(1)

    # --- Fase 3: Análisis Estocástico ---
    print("\n  [FASE 3] Análisis estocástico y detección de Timeouts...")
    try:
        resumen, df_ventanas = analyze(
            df_clean,
            equipo=equipo_nombre,
            plots_dir=plots_dir,
        )
        
        # --- Guardar resumen estadístico en CSV ---
        summary_path = os.path.join(processed_dir, f"{team_name}_resumen_timeouts.csv")
        df_summary = pd.DataFrame([resumen])
        df_summary.to_csv(summary_path, index=False)
        print(f"\n  Resumen numérico guardado en: {summary_path}")
        
    except Exception as e:
        print(f"\n  ERROR en el análisis estocástico: {e}")
        sys.exit(1)

    print(f"\n{'#'*60}")
    print(f"  PROYECTO EJECUTADO EXITOSAMENTE")
    print(f"  Gráficas guardadas en: {plots_dir}")
    print(f"  Datos limpios procesados en: {processed_dir}")
    print(f"{'#'*60}\n")

if __name__ == "__main__":
    main()
