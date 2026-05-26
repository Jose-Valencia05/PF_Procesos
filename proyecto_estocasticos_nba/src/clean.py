"""
Módulo de Limpieza y Procesamiento - Datos Play-by-Play V3
=========================================================
Procesa los datos jugada a jugada (PBP V3), calcula los tiempos de juego continuos,
separa los puntos anotados de los recibidos por minuto y genera series temporales
regulares por partido.

Autor: Proyecto Procesos Estocásticos - NBA
"""

import os
import pandas as pd
import numpy as np
import re

def parse_clock_v3(clock_str):
    """Convierte string de reloj ISO 'PTMMMSS.00S' a segundos totales."""
    if not isinstance(clock_str, str) or pd.isna(clock_str):
        return 0.0
    # clock_str tiene formato "PT12M00.00S" o "PT09M14.00S" o "PT05.00S"
    match = re.match(r'PT(?:(\d+)M)?(?:([\d.]+)S)?', clock_str)
    if match:
        min_val = float(match.group(1)) if match.group(1) else 0.0
        sec_val = float(match.group(2)) if match.group(2) else 0.0
        return min_val * 60.0 + sec_val
    return 0.0

def clean_pbp_data(pbp_path, metadata_path, team_name="warriors"):
    """
    Función de procesamiento principal para limpiar y transformar PBP V3 a series temporales por minuto.
    """
    print(f"  Cargando PBP V3 desde: {pbp_path}")
    print(f"  Cargando Metadatos desde: {metadata_path}")
    
    df_pbp = pd.read_csv(pbp_path)
    df_meta = pd.read_csv(metadata_path)
    
    # Asegurar tipos homogéneos (strings) para las llaves de cruce
    df_pbp['gameId'] = df_pbp['gameId'].astype(str)
    df_meta['GAME_ID'] = df_meta['GAME_ID'].astype(str)
    
    # 1. Parsear el tiempo transcurrido en minutos
    # clock contiene el tiempo restante en el período en formato ISO (ej: PT12M00.00S).
    print("  Calculando tiempos transcurridos...")
    df_pbp['remaining_seconds'] = df_pbp['clock'].apply(parse_clock_v3)
    
    def calculate_elapsed_minutes(row):
        period = int(row['period'])
        rem_sec = row['remaining_seconds']
        if period <= 4:
            # Cuarto regular (12 minutos = 720 segundos)
            elapsed_sec = (period - 1) * 720 + (720 - rem_sec)
        else:
            # Overtime (cada OT es de 5 min = 300 segundos)
            elapsed_sec = 4 * 720 + (period - 5) * 300 + (300 - rem_sec)
        return elapsed_sec / 60.0
    
    df_pbp['elapsed_minutes'] = df_pbp.apply(calculate_elapsed_minutes, axis=1)
    
    # 2. Parsear el marcador (scoreHome y scoreAway)
    # En la API V3, scoreHome y scoreAway vienen de forma nativa e independiente.
    print("  Separando marcadores...")
    df_pbp['scoreHome'] = pd.to_numeric(df_pbp['scoreHome'], errors='coerce').ffill().fillna(0).astype(int)
    df_pbp['scoreAway'] = pd.to_numeric(df_pbp['scoreAway'], errors='coerce').ffill().fillna(0).astype(int)
    
    # Calcular puntos anotados en cada jugada
    # Para evitar números negativos en revisiones de mesa, limitamos los cambios a >= 0
    df_pbp['home_pts_scored'] = df_pbp.groupby('gameId')['scoreHome'].diff().fillna(0).clip(lower=0).astype(int)
    df_pbp['away_pts_scored'] = df_pbp.groupby('gameId')['scoreAway'].diff().fillna(0).clip(lower=0).astype(int)
    
    # 3. Determinar si el equipo analizado es Home o Visitor en cada juego
    print("  Mapeando condición de Local/Visitante...")
    # En MATCHUP, 'vs.' indica LOCAL (Home), '@' indica VISITANTE (Visitor)
    meta_mapping = {}
    for _, row in df_meta.iterrows():
        game_id = str(row['GAME_ID'])
        matchup = row['MATCHUP']
        is_home = 'vs.' in matchup
        meta_mapping[game_id] = is_home
        
    df_pbp['is_home'] = df_pbp['gameId'].map(meta_mapping)
    
    # Puntos anotados y recibidos por el equipo analizado en cada jugada
    df_pbp['pts_scored'] = np.where(df_pbp['is_home'] == True, df_pbp['home_pts_scored'], df_pbp['away_pts_scored'])
    df_pbp['pts_received'] = np.where(df_pbp['is_home'] == True, df_pbp['away_pts_scored'], df_pbp['home_pts_scored'])
    
    # Renombrar gameId a GAME_ID para mantener compatibilidad
    df_pbp = df_pbp.rename(columns={'gameId': 'GAME_ID'})
    
    # 4. Agrupación temporal en minutos regulares (1 a 48+)
    print("  Creando grilla regular de minutos para el análisis temporal...")
    # Cada jugada cae en un minuto entero discreto (ej: del minuto 0.0 al 1.0 es el minuto 1)
    df_pbp['minute_bin'] = np.ceil(df_pbp['elapsed_minutes']).clip(lower=1).astype(int)
    
    # Resumir puntos anotados y recibidos por partido y por minuto
    minute_puntos = df_pbp.groupby(['GAME_ID', 'minute_bin'])[['pts_scored', 'pts_received']].sum().reset_index()
    
    # Reindexar cada partido para que tenga TODOS los minutos secuenciales (evitar vacíos)
    all_game_minutes = []
    for game_id, group in minute_puntos.groupby('GAME_ID'):
        max_min = max(48, group['minute_bin'].max())  # Asegurar al menos 48 minutos
        full_range = pd.DataFrame({'minute_bin': range(1, max_min + 1)})
        full_range['GAME_ID'] = game_id
        
        # Merge de los puntos reales con el rango completo de minutos
        game_full = pd.merge(full_range, group, on=['GAME_ID', 'minute_bin'], how='left').fillna(0)
        game_full['pts_scored'] = game_full['pts_scored'].astype(int)
        game_full['pts_received'] = game_full['pts_received'].astype(int)
        all_game_minutes.append(game_full)
        
    df_time_series = pd.concat(all_game_minutes, ignore_index=True)
    
    return df_time_series

def clean(raw_path=None, team_name="warriors", output_dir=None,
          project_root=None, seasons=None):
    """
    Funcion principal del modulo de limpieza.
    Ejecuta todo el pipeline de limpieza.

    Parameters
    ----------
    raw_path : str, optional
        Ruta al archivo CSV crudo. Si es None, se busca en data/raw/.
    team_name : str
        Nombre del equipo para localizar/guardar archivos.
    output_dir : str, optional
        Directorio donde guardar el CSV procesado.
    project_root : str, optional
        Directorio raiz del proyecto (para localizar el raw por defecto).
    seasons : list, optional
        Lista de temporadas en formato 'YYYY-YY' para filtrar los datos
        crudos antes de limpiar.

    Returns
    -------
    tuple
        (DataFrame limpio, ruta del archivo guardado)
    """
    print(f"\n{'='*60}")
    print(f"  LIMPIEZA Y PROCESAMIENTO PLAY-BY-PLAY")
    print(f"{'='*60}\n")
    
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
    raw_pbp_path = os.path.join(project_root, "data", "raw", f"{team_name}_pbp_raw.csv")
    raw_meta_path = os.path.join(project_root, "data", "raw", f"{team_name}_metadata_raw.csv")
    
    if not os.path.exists(raw_pbp_path) or not os.path.exists(raw_meta_path):
        raise FileNotFoundError(
            "Faltan archivos crudos en data/raw. Por favor ejecute extract.py primero."
        )
        
    df_clean = clean_pbp_data(raw_pbp_path, raw_meta_path, team_name=team_name)
    
    if output_dir is None:
        output_dir = os.path.join(project_root, "data", "processed")

    df = cargar_datos_crudos(raw_path)

    if seasons is not None:
        season_col = None
        for c in ["SEASON_YEAR", "season_year"]:
            if c in df.columns:
                season_col = c
                break
        if season_col:
            mask = df[season_col].isin(seasons)
            df = df[mask].copy()
            print(f"  Filtrado a temporadas {seasons}: {len(df)} registros")
        else:
            print("  Columna SEASON_YEAR no encontrada, sin filtro por temporada.")

    df = filtrar_columnas_relevantes(df)
    df = renombrar(df)
    df = limpiar_nulos(df)
    df = filtrar_temporada_regular(df)
    df = convertir_fecha(df)

    filepath = guardar_datos_procesados(df, team_name=team_name,
                                         processed_dir=output_dir)
    return df, filepath


if __name__ == "__main__":
    clean(team_name="warriors")
