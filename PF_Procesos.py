import pandas as pd
# Importamos el endpoint específico para registros de partidos por equipo
from nba_api.stats.endpoints import teamgamelogs
import time

def obtener_datos_nba():
    print("Iniciando conexión con el enrutador de la API de la NBA...")
    
    # 1. Definir el parámetro temporal
    # La temporada 2024-2025 se define con esta nomenclatura en la API
    temporada_objetivo = '2024-25'
    
    try:
        # 2. Realizar la petición al Endpoint
        # Esto extrae TODOS los partidos jugados por todos los equipos en esa temporada
        logs = teamgamelogs.TeamGameLogs(season_nullable=temporada_objetivo)
        
        # 3. Transformación del JSON a un DataFrame
        df_nba = logs.get_data_frames()[0]
        
        # 4. Filtrado de Variables Estocásticas
        # FG3M = Triples (3 Pointers Made)
        # REB = Rebotes Totales
        # AST = Asistencias
        # BLK = Tapones (Blocks)
        # PTS = Puntos totales
        columnas_objetivo = ['TEAM_NAME', 'GAME_DATE', 'MATCHUP', 'PTS', 'FG3M', 'REB', 'AST', 'BLK']
        
        # Seleccionamos solo las columnas que nos interesan
        df_filtrado = df_nba[columnas_objetivo]
        
        # 5. Guardado del Dataset
        nombre_archivo = 'dataset_nba_poisson_2025.csv'
        df_filtrado.to_csv(nombre_archivo, index=False)
        
        print(f"¡Éxito! Dataset guardado como '{nombre_archivo}'.")
        print(f"Total de registros (partidos) extraídos: {len(df_filtrado)}")
        
        # Mostramos una muestra de los primeros 5 registros
        print("\nMuestra de los datos:")
        print(df_filtrado.head())
        
    except Exception as e:
        print(f"Error en la extracción de datos: {e}")

# Ejecutar el script
if __name__ == "__main__":
    obtener_datos_nba()