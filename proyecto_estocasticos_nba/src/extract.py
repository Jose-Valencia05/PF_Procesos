"""
Módulo de Extracción de Datos
==============================
Utiliza nba_api para obtener datos históricos de partidos de la NBA.
Extrae los puntos anotados por partido (PTS) de un equipo específico
a lo largo de múltiples temporadas.

Autor: Proyecto Procesos Estocásticos - NBA
"""

import time
import os
import pandas as pd
from nba_api.stats.endpoints import teamgamelogs


TEAM_IDS = {
    "warriors": 1610612744,
    "lakers": 1610612747,
    "celtics": 1610612738,
    "bulls": 1610612741,
    "heat": 1610612748,
    "spurs": 1610612759,
    "nuggets": 1610612743,
    "bucks": 1610612749,
}


def get_seasons_list(start_year=2014, end_year=2025):
    """
    Genera la lista de temporadas en formato 'YYYY-YY'.

    Parameters
    ----------
    start_year : int
        Año de inicio de la primera temporada (ej. 2014 para 2014-15).
    end_year : int
        Año de inicio de la última temporada (ej. 2024 para 2024-25).

    Returns
    -------
    list
        Lista de strings con formato 'YYYY-YY'.
    """
    seasons = []
    for year in range(start_year, end_year + 1):
        season = f"{year}-{str(year + 1)[-2:]}"
        seasons.append(season)
    return seasons


def extract_team_game_logs(team_id=1610612744, seasons=None, sleep_time=0.6):
    """
    Extrae los game logs de un equipo para las temporadas especificadas.

    Realiza una solicitud HTTP a la API de NBA.com por cada temporada.
    Se incluye un retraso entre solicitudes para evitar rate limiting.

    Parameters
    ----------
    team_id : int
        ID del equipo en NBA.com. Default: 1610612744 (Warriors).
    seasons : list, optional
        Lista de temporadas en formato 'YYYY-YY'. Si es None,
        se usan las temporadas 2014-15 a 2024-25.
    sleep_time : float
        Segundos de espera entre solicitudes HTTP.

    Returns
    -------
    pandas.DataFrame or None
        DataFrame con todos los game logs combinados, o None si falla.
    """
    if seasons is None:
        seasons = get_seasons_list(2014, 2024)

    all_data = []
    failed_seasons = []

    print(f"\n{'='*60}")
    print(f"  EXTRACCIÓN DE DATOS - NBA API")
    print(f"  Team ID: {team_id}")
    print(f"  Temporadas: {len(seasons)} ({seasons[0]} a {seasons[-1]})")
    print(f"{'='*60}\n")

    for i, season in enumerate(seasons):
        try:
            print(f"  [{i+1}/{len(seasons)}] Temporada {season}...", end=" ")
            logs = teamgamelogs.TeamGameLogs(
                team_id_nullable=str(team_id),
                season_nullable=season,
                season_type_nullable="Regular Season",
            )
            df = logs.get_data_frames()[0]
            all_data.append(df)
            print(f"OK ({len(df)} partidos)")
            time.sleep(sleep_time)

        except Exception as e:
            print(f"FALLÓ: {e}")
            failed_seasons.append(season)
            time.sleep(sleep_time)

    if not all_data:
        print("\n  ERROR: No se pudo obtener datos de ninguna temporada.")
        return None

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n  Total de partidos extraídos: {len(combined)}")
    if failed_seasons:
        print(f"  Temporadas fallidas: {failed_seasons}")

    return combined


def save_raw_data(df, team_name="warriors", raw_dir=None):
    """
    Guarda el DataFrame crudo en formato CSV.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con los datos extraídos.
    team_name : str
        Nombre del equipo para el nombre del archivo.
    raw_dir : str, optional
        Directorio donde guardar el CSV.

    Returns
    -------
    str
        Ruta del archivo guardado.
    """
    if raw_dir is None:
        raw_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "raw"
        )

    os.makedirs(raw_dir, exist_ok=True)
    filepath = os.path.join(raw_dir, f"{team_name}_game_logs_raw.csv")
    df.to_csv(filepath, index=False)
    print(f"  Datos crudos guardados en: {filepath}")
    return filepath


def extract(team_name="warriors", seasons=None, output_dir=None):
    """
    Función principal del módulo de extracción.
    Orquesta la obtención y guardado de datos.

    Parameters
    ----------
    team_name : str
        Nombre del equipo (key de TEAM_IDS). Default: 'warriors'.
    seasons : list, optional
        Lista de temporadas en formato 'YYYY-YY'.
    output_dir : str, optional
        Directorio raíz del proyecto.

    Returns
    -------
    tuple
        (DataFrame con los datos, ruta del archivo guardado)
    """
    team_id = TEAM_IDS.get(team_name.lower(), 1610612744)

    df = extract_team_game_logs(team_id=team_id, seasons=seasons)

    if df is None:
        raise RuntimeError("La extracción de datos falló.")

    filepath = save_raw_data(df, team_name=team_name, raw_dir=output_dir)
    return df, filepath


if __name__ == "__main__":
    df, path = extract(team_name="warriors")
    print(f"\n Primeras 5 filas:\n{df.head()}")
    print(f"\n Columnas disponibles:\n{df.columns.tolist()}")
