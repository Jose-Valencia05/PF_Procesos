"""
Módulo de Limpieza y Preparación de Datos
==========================================
Lee los datos crudos extraídos de nba_api, limpia valores nulos,
filtra columnas relevantes, renombra variables al español y
guarda el dataset procesado listo para el análisis.

Autor: Proyecto Procesos Estocásticos - NBA
"""

import os
import pandas as pd
import numpy as np


COLUMNAS_RELEVANTES = [
    "TEAM_ID", "TEAM_NAME", "GAME_ID", "GAME_DATE",
    "MATCHUP", "WL", "PTS", "FGM", "FGA",
    "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT", "AST", "REB",
    "TOV", "PLUS_MINUS",
]


RENOMBRAR_COLUMNAS = {
    "TEAM_ID": "id_equipo",
    "TEAM_NAME": "equipo",
    "GAME_ID": "id_partido",
    "GAME_DATE": "fecha",
    "MATCHUP": "enfrentamiento",
    "WL": "resultado",
    "PTS": "puntos",
    "FGM": "tiros_campo_convertidos",
    "FGA": "tiros_campo_intentados",
    "FG_PCT": "porcentaje_tiros_campo",
    "FG3M": "triples_convertidos",
    "FG3A": "triples_intentados",
    "FG3_PCT": "porcentaje_triples",
    "FTM": "libres_convertidos",
    "FTA": "libres_intentados",
    "FT_PCT": "porcentaje_libres",
    "AST": "asistencias",
    "REB": "rebotes",
    "TOV": "perdidas",
    "PLUS_MINUS": "mas_menos",
}


def cargar_datos_crudos(filepath):
    """
    Carga el dataset crudo desde un archivo CSV.

    Parameters
    ----------
    filepath : str
        Ruta al archivo CSV con los datos crudos.

    Returns
    -------
    pandas.DataFrame
        DataFrame con los datos crudos.
    """
    print(f"  Cargando datos desde: {filepath}")
    df = pd.read_csv(filepath)
    print(f"  Registros cargados: {len(df)}")
    return df


def filtrar_columnas_relevantes(df):
    """
    Conserva solo las columnas definidas como relevantes para el análisis.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame original con todas las columnas.

    Returns
    -------
    pandas.DataFrame
        DataFrame filtrado con columnas relevantes.
    """
    columnas_presentes = [c for c in COLUMNAS_RELEVANTES if c in df.columns]
    faltantes = [c for c in COLUMNAS_RELEVANTES if c not in df.columns]
    if faltantes:
        print(f"  Columnas no encontradas (se ignoran): {faltantes}")
    return df[columnas_presentes].copy()


def renombrar(df):
    """
    Renombra las columnas del DataFrame a español.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con nombres de columna en inglés.

    Returns
    -------
    pandas.DataFrame
        DataFrame con nombres de columna en español.
    """
    mapping = {k: v for k, v in RENOMBRAR_COLUMNAS.items() if k in df.columns}
    return df.rename(columns=mapping)


def limpiar_nulos(df):
    """
    Elimina filas con valores nulos en las columnas numéricas clave.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con posibles valores nulos.

    Returns
    -------
    pandas.DataFrame
        DataFrame sin valores nulos en columnas numéricas.
    """
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    n_antes = len(df)
    df = df.dropna(subset=columnas_numericas)
    n_despues = len(df)
    eliminados = n_antes - n_despues
    if eliminados > 0:
        print(f"  Filas eliminadas por valores nulos: {eliminados}")
    return df


def filtrar_temporada_regular(df):
    """
    Filtra solo partidos de temporada regular si la columna existe.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con columna de tipo de temporada.

    Returns
    -------
    pandas.DataFrame
        DataFrame filtrado (o sin cambios si la columna no existe).
    """
    col = None
    for c in ["SEASON_TYPE", "season_type"]:
        if c in df.columns:
            col = c
            break
    if col:
        mask = df[col] == "Regular Season"
        df = df[mask].copy()
    return df


def convertir_fecha(df):
    """
    Convierte la columna de fecha a datetime y la ordena cronológicamente.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con columna 'fecha' (string).

    Returns
    -------
    pandas.DataFrame
        DataFrame ordenado por fecha.
    """
    col = "fecha" if "fecha" in df.columns else "GAME_DATE"
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df = df.sort_values(by=col).reset_index(drop=True)
    return df


def guardar_datos_procesados(df, team_name="warriors", processed_dir=None):
    """
    Guarda el DataFrame limpio en formato CSV.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame procesado.
    team_name : str
        Nombre del equipo para el nombre del archivo.
    processed_dir : str, optional
        Directorio donde guardar el archivo.

    Returns
    -------
    str
        Ruta del archivo guardado.
    """
    if processed_dir is None:
        processed_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "processed"
        )

    os.makedirs(processed_dir, exist_ok=True)
    filepath = os.path.join(processed_dir, f"{team_name}_game_logs_clean.csv")
    df.to_csv(filepath, index=False)
    print(f"  Datos procesados guardados en: {filepath}")
    return filepath


def clean(raw_path=None, team_name="warriors", output_dir=None, project_root=None):
    """
    Función principal del módulo de limpieza.
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
        Directorio raíz del proyecto (para localizar el raw por defecto).

    Returns
    -------
    tuple
        (DataFrame limpio, ruta del archivo guardado)
    """
    print(f"\n{'='*60}")
    print(f"  LIMPIEZA Y PREPARACIÓN DE DATOS")
    print(f"{'='*60}\n")

    if raw_path is None:
        if project_root:
            raw_path = os.path.join(project_root, "data", "raw",
                                    f"{team_name}_game_logs_raw.csv")
        else:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            raw_path = os.path.join(base, "data", "raw",
                                    f"{team_name}_game_logs_raw.csv")

    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"No se encontró el archivo de datos crudos: {raw_path}\n"
            f"Ejecute primero el módulo de extracción (extract.py)."
        )

    if output_dir is None and project_root:
        output_dir = os.path.join(project_root, "data", "processed")

    df = cargar_datos_crudos(raw_path)
    df = filtrar_columnas_relevantes(df)
    df = renombrar(df)
    df = limpiar_nulos(df)
    df = filtrar_temporada_regular(df)
    df = convertir_fecha(df)

    filepath = guardar_datos_procesados(df, team_name=team_name,
                                         processed_dir=output_dir)
    return df, filepath


if __name__ == "__main__":
    df, path = clean(team_name="warriors")
    print(f"\n Columnas finales: {df.columns.tolist()}")
    print(f"\n Estadísticas descriptivas de PTS:\n{df['puntos'].describe()}")
