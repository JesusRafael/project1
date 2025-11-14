import pandas as pd
import psycopg2
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ruptures as rpt
import time
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
import os

# Page configuration
st.set_page_config(
    page_title="Detector de Anomalías Climáticas Diarias",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database configuration - Secure credentials using Streamlit secrets or environment variables
# Priority: 1) Streamlit secrets, 2) Environment variables, 3) Default values for development
try:
    # Try to get from Streamlit secrets first (works in Streamlit Cloud and local with .streamlit/secrets.toml)
    DB_CONFIG = {
        'host': st.secrets.get("database", {}).get("host", os.environ.get('DB_HOST', 'YOUR_SERVER.postgres.database.azure.com')),
        'database': st.secrets.get("database", {}).get("database", os.environ.get('DB_NAME', 'stream')),
        'user': st.secrets.get("database", {}).get("user", os.environ.get('DB_USER', 'your_username')),
        'password': st.secrets.get("database", {}).get("password", os.environ.get('DB_PASSWORD', '')),
        'port': st.secrets.get("database", {}).get("port", os.environ.get('DB_PORT', '5432')),
        'sslmode': st.secrets.get("database", {}).get("sslmode", 'require')
    }
except Exception as e:
    # Fallback to environment variables if secrets not available
    DB_CONFIG = {
        'host': os.environ.get('DB_HOST', 'localhost'),
        'database': os.environ.get('DB_NAME', 'project1'),
        'user': os.environ.get('DB_USER', 'postgres'),
        'password': os.environ.get('DB_PASSWORD', '12345'),
        'port': os.environ.get('DB_PORT', '5432')
    }

st.title("Detector de Anomalías Climáticas Diarias 📊")

st.sidebar.header("Menú Principal")

selection = st.sidebar.radio(
    "Selecciona una sección:",
    ("1. Exploración de Datos (EDA)", "2. Detección de Anomalías", "3. Online")
)

st.sidebar.markdown("---")
st.sidebar.info("Utiliza esta barra lateral para navegar entre las secciones.")


@st.cache_resource
def get_db_connection():
    """Create and cache database connection"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None
    
def fetch_data(query, params=None):
    """Execute query and return DataFrame - NOT cached to get fresh data"""
    conn = get_db_connection()
    if conn:
        try:
            # Ensure connection is still alive
            conn.isolation_level
            df = pd.read_sql_query(query, conn, params=params)
            return df
        except (psycopg2.InterfaceError, psycopg2.OperationalError):
            # Connection lost, clear cache and reconnect
            st.cache_resource.clear()
            conn = get_db_connection()
            if conn:
                df = pd.read_sql_query(query, conn, params=params)
                return df
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Query error: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def detect_anomalies(df, column='reading_value', threshold=2.5):
    """Simple anomaly detection using z-score"""
    if len(df) > 0:
        mean = df[column].mean()
        std = df[column].std()
        df['z_score'] = (df[column] - mean) / std if std > 0 else 0
        df['is_anomaly'] = df['z_score'].abs() > threshold
        return df
    return df
sql_daily = f"""
    SELECT * FROM datos.daily;
    """

dfd = fetch_data(sql_daily)
dfd = dfd.iloc[11:].copy()
if 'time' in dfd.columns:
    dfd['time'] = pd.to_datetime(dfd['time'])
    dfd = dfd.set_index('time')

variable_estudio = 'precipitation_probability_mean'




df_prophet = dfd[[variable_estudio]].reset_index()
df_prophet.rename(columns={'time': 'ds', variable_estudio: 'y'}, inplace=True)
df_prophet['ds'] = df_prophet['ds'].dt.date
model = Prophet(
    yearly_seasonality=False, 
    weekly_seasonality=True,  
    daily_seasonality=False
)
model.fit(df_prophet)

# 2. Predecir
future = model.make_future_dataframe(periods=0)
forecast = model.predict(future)

# 3. Unir los resultados con el DataFrame original
df_anomalia = df_prophet.set_index('ds').join(forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']])
df_anomalia['anomaly'] = 0
df_anomalia.loc[df_anomalia['y'] > df_anomalia['yhat_upper'], 'anomaly'] = 1 
df_anomalia.loc[df_anomalia['y'] < df_anomalia['yhat_lower'], 'anomaly'] = -1 

dias_anomalos = df_anomalia[df_anomalia['anomaly'] != 0].copy()

if selection == "1. Exploración de Datos (EDA)":
    ## 📈 Exploración de Datos (EDA)
    st.header("1. Exploración de Datos (EDA) 🔎")
    st.write("Esta sección es para el Análisis Exploratorio de Datos.")
    
    st.subheader("Gráfica 1: Título de la Gráfica 1")
    
    
    dfd[variable_estudio] = pd.to_numeric(dfd[variable_estudio], errors='coerce')
    dfd = dfd.dropna(subset=[variable_estudio])


    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # Histograma
    sns.histplot(dfd[variable_estudio], kde=True, ax=ax[0])
    ax[0].set_title(f'Distribución de {variable_estudio}', fontsize=14)
    ax[0].set_xlabel(variable_estudio)
    ax[0].set_ylabel('Frecuencia')

    # Boxplot
    sns.boxplot(y=dfd[variable_estudio], ax=ax[1])
    ax[1].set_title(f'Boxplot de {variable_estudio}', fontsize=14)
    ax[1].set_ylabel(variable_estudio)
    
    plt.tight_layout()
    st.pyplot(fig)
#  ----------------------------------------------------------------------------------------------------------------- 
    
    st.markdown("---")
    
    st.subheader(f"Gráfica 2: Serie de Tiempo de {variable_estudio}")
    fig_ts, ax_ts = plt.subplots(figsize=(14, 6))
    
    dfd[variable_estudio].plot(
        title=f'Serie de Tiempo: {variable_estudio}',
        ax=ax_ts
    )
    ax_ts.set_xlabel('Fecha')
    ax_ts.set_ylabel('Valor') 
    ax_ts.grid(True)
    st.pyplot(fig_ts)

    st.markdown("---")

    st.subheader(f"Gráfica 3: Descomposición de Serie de Tiempo ({variable_estudio})")
    decomposition = seasonal_decompose(dfd[variable_estudio], model='additive', period=7)
    decomposition = seasonal_decompose(dfd[variable_estudio], model='additive', period=7) # Asumiendo estacionalidad anual

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    decomposition.observed.plot(ax=ax1, title='Observado')
    decomposition.trend.plot(ax=ax2, title='Tendencia')
    decomposition.seasonal.plot(ax=ax3, title='Estacionalidad')
    decomposition.resid.plot(ax=ax4, title='Residuos (Anomalías)')
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")




elif selection == "2. Detección de Anomalías":
    
    st.header("2. Detección de Anomalías 🚨")
    st.write("Esta sección se centrará en los resultados del modelo de detección de anomalías.")
    st.subheader("Gráfica A: Distribución de Puntuaciones de Anomalía")


    st.info(f"\nTotal de días anómalos encontrados: {len(dias_anomalos)}")
    
    fig = plt.figure(figsize=(16, 7))
    plt.plot(df_anomalia.index, df_anomalia['y'], label='Valor Real', color='blue')
    plt.plot(df_anomalia.index, df_anomalia['yhat'], label='Predicción (Modelo Prophet)', color='green', linestyle='--')

    # 2. Sombrear la banda de confianza
    plt.fill_between(df_anomalia.index, df_anomalia['yhat_lower'], df_anomalia['yhat_upper'], 
                    color='green', alpha=0.1, label='Banda de Confianza')

    # 3. Marcar los puntos anómalos
    plt.scatter(dias_anomalos.index, dias_anomalos['y'], color='red', s=40, label='Día Anómalo')

    plt.title(f'Detección de Días Anómalos en {variable_estudio} con Prophet')
    plt.xlabel('Fecha')
    plt.ylabel('Probabilidad de Precipitación (%)')
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)


    st.markdown("---")

    # st.subheader("Gráfica B: Serie de Tiempo con Anomalías Marcadas")
    # # Ejemplo de placeholder de gráfica
    # st.warning("Espacio para la segunda gráfica de anomalías (e.g., serie de tiempo con puntos anómalos resaltados).")
    
    
elif selection == "3. Online":
    ## 🌐 Online
    st.header("3. Online 🌐")
    st.write("Esta sección está reservada para el despliegue o la monitorización en tiempo real (Online).")
    
    # --- Selectbox para el Intervalo de Monitorización ---
    monitor_interval = st.selectbox(
        "Selecciona el intervalo de monitorización (Ventana de datos):",
        options=["5 minutos", "30 minutos", "1 hora", "4 horas", "24 horas"],
        index=0 
    )
    
    st.info("Contenido pendiente. Se implementará más adelante.")