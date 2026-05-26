"""
DASHBOARD REACTIVO - PANEL DE CONTROL TÁCTICO NBA POISSON
=========================================================
Proporciona una interfaz web interactiva usando Streamlit. Permite a los usuarios:
1. Seleccionar cualquiera de los 30 equipos de la NBA.
2. Descargar datos Play-by-Play de la API en tiempo real o usar la caché local.
3. Ajustar de forma interactiva los hiperparámetros estocásticos (t, alpha, cooldown).
4. Visualizar curvas de puntuación acumulada y tiempos fuera recomendados (🔥).
5. Analizar el Índice de Dispersión y justificar el tamaño de ventana óptimo.

Uso:
    streamlit run dashboard.py
"""

import os
import sys
import pandas as pd
import streamlit as st

# Insertar el root del proyecto en el path para asegurar la importación de src
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Importar funciones clave del motor analítico
from src.extract import TEAM_IDS, extract
from src.clean import clean
from src.analysis import evaluar_ventanas, detectar_time_outs, graficar_partido_con_timeouts

# Configuración premium de Streamlit
st.set_page_config(
    page_title="NBA Poisson Timeout Dashboard",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para dar una estética oscura y deportiva ultra-premium
st.markdown("""
    <style>
    .main {
        background-color: #fafafa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
    }
    .highlight-card {
        background-color: #fff8f8;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #d62728;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Mapeo de nombres completos de equipos para el selector de la UI
TEAM_DISPLAY_NAMES = {
    "Atlanta Hawks": "hawks",
    "Boston Celtics": "celtics",
    "Brooklyn Nets": "nets",
    "Charlotte Hornets": "hornets",
    "Chicago Bulls": "bulls",
    "Cleveland Cavaliers": "cavaliers",
    "Dallas Mavericks": "mavericks",
    "Denver Nuggets": "nuggets",
    "Detroit Pistons": "pistons",
    "Golden State Warriors": "warriors",
    "Houston Rockets": "rockets",
    "Indiana Pacers": "pacers",
    "Los Angeles Clippers": "clippers",
    "Los Angeles Lakers": "lakers",
    "Memphis Grizzlies": "grizzlies",
    "Miami Heat": "heat",
    "Milwaukee Bucks": "bucks",
    "Minnesota Timberwolves": "timberwolves",
    "New Orleans Pelicans": "pelicans",
    "New York Knicks": "knicks",
    "Oklahoma City Thunder": "thunder",
    "Orlando Magic": "magic",
    "Philadelphia 76ers": "76ers",
    "Phoenix Suns": "suns",
    "Portland Trail Blazers": "blazers",
    "Sacramento Kings": "kings",
    "San Antonio Spurs": "spurs",
    "Toronto Raptors": "raptors",
    "Jazz": "jazz",
    "Washington Wizards": "wizards",
}

# --- TÍTULO PRINCIPAL ---
st.title("🏀 NBA-Poisson-TimeOut: Panel de Control Táctico")
st.markdown("---")

# --- CONTROLADORES EN LA BARRA LATERAL ---
st.sidebar.image("https://img.icons8.com/color/96/000000/basketball.png", width=80)
st.sidebar.header("Parámetros del Proceso")

# 1. Selector de Equipo
team_display = st.sidebar.selectbox("Selecciona la Franquicia a Auditar:", list(TEAM_DISPLAY_NAMES.keys()), index=9)
team_key = TEAM_DISPLAY_NAMES[team_display]

# 2. Control de Datos de API
st.sidebar.subheader("Descarga e Historial")
limit_games = st.sidebar.slider("Límite de partidos a procesar:", min_value=1, max_value=15, value=5)
force_download = st.sidebar.checkbox("Forzar descarga de la API (nba_api)", value=False, 
                                     help="Si se desmarca, se usarán los archivos CSV guardados localmente si existen.")

# 3. Ajuste de Parámetros Estocásticos
st.sidebar.subheader("Hiperparámetros de Poisson")
window_t = st.sidebar.slider("Ventana de tiempo móvil t (minutos):", min_value=2, max_value=5, value=3, 
                             help="Tamaño de la ventana móvil para acumular los puntos del rival.")
alpha_threshold = st.sidebar.slider("Nivel de significancia crítica α (p-valor):", min_value=0.01, max_value=0.10, value=0.05, step=0.01,
                                    help="Umbral de probabilidad acumulada por debajo del cual se considera racha anómala.")
cooldown_min = st.sidebar.slider("Cooldown de Tiempos Fuera (minutos):", min_value=2, max_value=8, value=4,
                                 help="Distancia mínima de juego requerida entre dos alertas de tiempo fuera consecutivas.")

# Botón Principal
ejecutar = st.sidebar.button("📊 EJECUTAR ANÁLISIS ESTOCÁSTICO", type="primary")

# Carpetas de datos locales
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Lógica del pipeline
pbp_path = os.path.join(RAW_DIR, f"{team_key}_pbp_raw.csv")
metadata_path = os.path.join(RAW_DIR, f"{team_key}_metadata_raw.csv")

# Intentar cargar datos si ya existen
data_loaded = False
if not ejecutar and os.path.exists(pbp_path) and os.path.exists(metadata_path):
    # Cargar los datos existentes para no forzar la ejecución si no se ha presionado el botón
    data_loaded = True

if ejecutar or data_loaded:
    try:
        # --- FASE 1: EXTRACCIÓN ---
        if ejecutar and force_download:
            with st.spinner(f"Descargando Play-by-Play de los {team_display} desde NBA.com..."):
                extract(team_name=team_key, limit_games=limit_games, output_dir=RAW_DIR)
                st.sidebar.success("Datos descargados correctamente.")
        elif ejecutar and not os.path.exists(pbp_path):
            with st.spinner(f"No hay caché local. Descargando datos iniciales desde la API..."):
                extract(team_name=team_key, limit_games=limit_games, output_dir=RAW_DIR)
                st.sidebar.success("Datos descargados correctamente.")
        
        # --- FASE 2: LIMPIEZA ---
        with st.spinner("Procesando Play-by-Play y construyendo series de tiempo..."):
            df_time_series, clean_path = clean(team_name=team_key, project_root=PROJECT_ROOT)
        
        # --- FASE 3: ANÁLISIS ESTOCÁSTICO DINÁMICO ---
        # 1. Calcular la tasa base
        lambda_minuto = df_time_series['pts_received'].mean()
        
        # 2. Evaluar ventanas móviles dinámicamente
        df_ventanas_rolling = []
        ventanas = [2, 3, 4, 5]
        for v in ventanas:
            df_rolling = df_time_series.groupby('GAME_ID')['pts_received'].rolling(window=v, min_periods=v).sum().reset_index(0, drop=True)
            df_rolling = df_rolling.dropna()
            df_ventanas_rolling.append({
                "Ventana (min)": v,
                "Lambda Empírico": df_rolling.mean(),
                "Lambda Teórico (t * λ)": lambda_minuto * v,
                "Varianza Empírica": df_rolling.var(),
                "Ratio Dispersión (Var/Media)": df_rolling.var() / df_rolling.mean() if df_rolling.mean() > 0 else 1.0
            })
        df_ventanas_df = pd.DataFrame(df_ventanas_rolling)
        
        # 3. Encontrar el mejor partido de showcase (que tenga al menos una sugerencia de timeout con los parámetros del slider)
        game_ids = df_time_series['GAME_ID'].unique()
        game_id_showcase = game_ids[0]
        
        for gid in game_ids:
            df_g = df_time_series[df_time_series['GAME_ID'] == gid].copy()
            df_g_analizado = detectar_time_outs(df_g, lambda_minuto, t=window_t, alpha=alpha_threshold, cooldown_min=cooldown_min)
            if df_g_analizado['sugerir_timeout'].sum() > 0:
                game_id_showcase = gid
                break
                
        # 4. Analizar el partido seleccionado
        df_game_showcase = df_time_series[df_time_series['GAME_ID'] == game_id_showcase].copy()
        df_game_analizado = detectar_time_outs(df_game_showcase, lambda_minuto, t=window_t, alpha=alpha_threshold, cooldown_min=cooldown_min)
        
        # 5. Generar la gráfica premium correspondiente
        plot_path = graficar_partido_con_timeouts(df_game_analizado, t=window_t, alpha=alpha_threshold, plots_dir=PLOTS_DIR)
        
        # --- RENDERIZAR INTERFAZ DE STREAMLIT ---
        
        # KPIs en la parte superior
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tasa Base Rival (λ)", f"{lambda_minuto:.3f} pts/min", help="Puntos promedio concedidos al rival por cada minuto de juego regular.")
        with col2:
            st.metric("Timeouts Sugeridos (🔥)", f"{int(df_game_analizado['sugerir_timeout'].sum())}", help="Número de alertas críticas de timeout generadas en el partido bajo el nivel de significancia seleccionado.")
        with col3:
            max_racha = df_game_analizado['pts_received_rolling'].max()
            st.metric("Max Racha del Rival", f"{int(max_racha) if not pd.isna(max_racha) else 0} pts", f"en {window_t} minutos")
        with col4:
            st.metric("Partido Showcase", f"ID: {game_id_showcase}", help="Partido representativo analizado. El sistema selecciona automáticamente el encuentro con mayor actividad táctica.")

        # Tabs principales de visualización
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Visualización Táctica", 
            "📋 Registro de Tiempos Críticos", 
            "🧮 Auditoría Matemática", 
            "⏱️ Línea de Tiempo Completa"
        ])
        
        # TAB 1: VISUALIZACIÓN TÁCTICA
        with tab1:
            st.subheader("Análisis Dinámico de Curvas de Puntuación")
            st.image(plot_path, use_column_width=True)
            st.info("💡 **Cómo leer este gráfico**: La línea naranja muestra la puntuación del rival. Las líneas verticales rojas marcan el instante preciso en el que el modelo Poisson de cola derecha detecta que la racha del oponente tiene una probabilidad menor a tu umbral crítico seleccionado (se rechaza la hipótesis nula). El entrenador debió pedir el tiempo fuera ahí.")
            
        # TAB 2: REGISTRO DE TIEMPOS CRÍTICOS
        with tab2:
            st.subheader("Instantes Exactos de Alerta Estocástica")
            timeouts_df = df_game_analizado[df_game_analizado['sugerir_timeout'] == True].copy()
            if not timeouts_df.empty:
                timeouts_table = timeouts_df[[
                    "minute_bin", "pts_received", "pts_received_rolling", "poisson_prob"
                ]].rename(columns={
                    "minute_bin": "Minuto del Juego",
                    "pts_received": "Puntos del Rival (Minuto Actual)",
                    "pts_received_rolling": f"Racha Total Acumulada ({window_t} min)",
                    "poisson_prob": "p-valor (Poisson)"
                })
                
                # Resaltar en una tabla interactiva
                st.dataframe(timeouts_table.style.format({"p-valor (Poisson)": "{:.5f}"}), use_container_width=True)
                
                for _, row in timeouts_df.iterrows():
                    st.markdown(f"""
                    <div class="highlight-card">
                        <strong>🔥 ALERTA CRÍTICA - Minuto {int(row['minute_bin'])}</strong><br/>
                        El rival anotó un acumulado de <b>{int(row['pts_received_rolling'])} puntos</b> en los últimos {window_t} minutos de juego.
                        La probabilidad estocástica de este evento es de solo <b>{row['poisson_prob']:.5f}</b> (menor al umbral crítico del {alpha_threshold * 100}%).
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ **Comportamiento Estable**: No se registraron anomalías críticas de puntuación ni rachas rivales significativas en este partido para los parámetros seleccionados.")
                
        # TAB 3: AUDITORÍA MATEMÁTICA
        with tab3:
            st.subheader("Análisis del Índice de Dispersión (Fisher)")
            st.write("El Índice de Dispersión ($D = \\sigma^2 / \\mu$) indica el comportamiento estocástico del flujo de puntos a diferentes escalas. Si $D \\approx 1$, el proceso es perfectamente modelable por Poisson. Si $D > 1$, hay sobredispersión (ruido), y si $D < 1$ hay subdispersión (homogeneidad estructural).")
            
            # Formatear la tabla del resumen de ventanas móviles
            st.dataframe(df_ventanas_df.style.format({
                "Lambda Empírico": "{:.4f}",
                "Lambda Teórico (t * λ)": "{:.4f}",
                "Varianza Empírica": "{:.4f}",
                "Ratio Dispersión (Var/Media)": "{:.4f}"
            }), use_container_width=True)
            
            # Explicación dinámica de la ventana óptima calculada para este equipo
            optimo_t_df = df_ventanas_df.iloc[(df_ventanas_df['Ratio Dispersión (Var/Media)'] - 1).abs().argsort()[:1]]
            t_recomendado = optimo_t_df['Ventana (min)'].values[0]
            
            st.success(f"🎯 **Sugerencia de Ingeniería de Datos**: La ventana empírica que mejor se ajusta a las propiedades puras del proceso de Poisson para los **{team_display}** es de **t = {t_recomendado} minutos** (con un Índice de Dispersión de **{optimo_t_df['Ratio Dispersión (Var/Media)'].values[0]:.4f}**, el más cercano a 1).")
            
        # TAB 4: LÍNEA DE TIEMPO COMPLETA
        with tab4:
            st.subheader("Registro de Puntuación Minuto a Minuto")
            timeline_table = df_game_analizado[[
                "minute_bin", "pts_scored", "pts_received", "pts_received_rolling", "poisson_prob", "sugerir_timeout"
            ]].rename(columns={
                "minute_bin": "Minuto",
                "pts_scored": "Nuestros Puntos",
                "pts_received": "Puntos del Rival",
                "pts_received_rolling": f"Racha Rival ({window_t} min)",
                "poisson_prob": "p-valor (Poisson)",
                "sugerir_timeout": "Señal Timeout"
            })
            st.dataframe(timeline_table.style.format({"p-valor (Poisson)": "{:.5f}"}), use_container_width=True)
            
    except Exception as e:
        st.error(f"Ocurrió un error en la ejecución de la tubería de datos: {e}")
        st.info("Sugerencia: Si es un error de la API de la NBA (Rate Limit), por favor desmarca la opción 'Forzar descarga' para trabajar con los datos almacenados en la caché local.")
else:
    # Estado inicial cuando se abre el dashboard y no hay caché de datos
    st.info(f"👋 **Bienvenido al Panel Táctico**. Selecciona un equipo en la barra lateral izquierda y presiona el botón **EJECUTAR ANÁLISIS ESTOCÁSTICO** para cargar el modelo por primera vez.")
    st.image("plots/timeout_analysis.png", width=600, caption="Ejemplo de la visualización de Tiempos Fuera esperada.")
