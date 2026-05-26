"""
Módulo de Extracción de Datos - Jugada a Jugada (Play-by-Play V3)
=================================================================
Utiliza nba_api para obtener datos jugada a jugada (PBP V3) de la temporada 2024-25.
La API de la NBA requiere el uso de PlayByPlayV3 para la temporada 2024-25,
ya que la versión anterior (V2) ha sido discontinuada y retorna respuestas vacías.
Implementa reintentos automáticos y retrasos adaptativos para manejar el Rate Limit.

Autor: Proyecto Procesos Estocásticos - NBA
"""

import time
import os
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, playbyplayv3

# Diccionario de IDs oficiales de la NBA para los equipos principales
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

def extract_2024_25_games(team_id, sleep_time=1.0):
    """
    Busca todos los partidos de la temporada 2024-25 para un equipo en particular.
    """
    print(f"  Buscando partidos de la temporada 2024-25 para Team ID: {team_id}...")
    try:
        finder = leaguegamefinder.LeagueGameFinder(
            team_id_nullable=str(team_id),
            season_nullable="2024-25",
            league_id_nullable="00"  # 00 representa la NBA regular
        )
        df_games = finder.get_data_frames()[0]
        # Filtrar solo juegos de temporada regular
        df_games = df_games[df_games["SEASON_ID"].astype(str).str.endswith("2024")].copy()
        print(f"  Partidos encontrados: {len(df_games)}")
        time.sleep(sleep_time)
        return df_games
    except Exception as e:
        print(f"  ERROR al obtener lista de partidos: {e}")
        return None

def extract_game_pbp(game_id, max_retries=5, base_delay=2.0):
    """
    Extrae los datos Play-by-Play V3 de un partido específico con reintentos y exponencial backoff.
    """
    delay = base_delay
    for attempt in range(max_retries):
        try:
            # Usar PlayByPlayV3 de nba_api (la V2 retorna un JSON vacío en temporadas recientes)
            pbp = playbyplayv3.PlayByPlayV3(game_id=str(game_id))
            df_pbp = pbp.get_data_frames()[0]
            if df_pbp is not None and not df_pbp.empty:
                return df_pbp
            print(f"    [Intento {attempt+1}/{max_retries}] Petición vacía para Game ID {game_id}. Reintentando...")
        except Exception as e:
            print(f"    [Intento {attempt+1}/{max_retries}] Error en Game ID {game_id}: {e}. Esperando {delay:.2f}s...")
        time.sleep(delay)
        delay *= 1.5  # Exponencial Backoff
    
    print(f"    [FALLO] No se pudo obtener PBP para el partido {game_id} después de {max_retries} intentos.")
    return None

def extract(team_name="warriors", limit_games=5, output_dir=None):
    """
    Orquesta la extracción de partidos y sus respectivos Play-by-Play.
    """
    team_name = team_name.lower()
    team_id = TEAM_IDS.get(team_name, 1610612744)

    print(f"\n{'='*60}")
    print(f"  NUEVA EXTRACCIÓN PLAY-BY-PLAY (V3) - TEMPORADA 2024-25")
    print(f"  Equipo: {team_name.upper()} (ID: {team_id})")
    print(f"  Límite de partidos a procesar: {limit_games}")
    print(f"{'='*60}\n")

    # 1. Obtener la lista de partidos de la temporada
    df_games = extract_2024_25_games(team_id)
    if df_games is None or df_games.empty:
        raise RuntimeError("No se pudieron obtener los partidos de la temporada 2024-25.")

    # Guardar metadatos de los partidos para la fase de limpieza
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "raw"
        )
    os.makedirs(output_dir, exist_ok=True)

    metadata_path = os.path.join(output_dir, f"{team_name}_metadata_raw.csv")
    df_games.to_csv(metadata_path, index=False)
    print(f"  Metadatos de partidos guardados en: {metadata_path}")

    # Ordenar cronológicamente y tomar los últimos limit_games partidos
    df_games = df_games.sort_values(by="GAME_DATE").tail(limit_games)
    game_ids = df_games["GAME_ID"].tolist()

    all_pbps = []
    print(f"\n  Iniciando descarga secuencial de Play-by-Play (V3)...")
    for idx, game_id in enumerate(game_ids):
        print(f"  [{idx+1}/{len(game_ids)}] Descargando PBP para Game ID: {game_id}...")
        df_pbp = extract_game_pbp(game_id)
        if df_pbp is not None:
            all_pbps.append(df_pbp)
        # Sleep preventivo entre partidos para respetar el Rate Limit
        time.sleep(1.5)

    if not all_pbps:
        raise RuntimeError("No se pudo obtener el PBP de ningún partido.")

    combined_pbp = pd.concat(all_pbps, ignore_index=True)
    pbp_path = os.path.join(output_dir, f"{team_name}_pbp_raw.csv")
    combined_pbp.to_csv(pbp_path, index=False)
    print(f"\n  ¡Éxito! Datos Play-by-Play (V3) guardados en: {pbp_path}")
    
    return combined_pbp, pbp_path

if __name__ == "__main__":
    extract(team_name="warriors", limit_games=2)
