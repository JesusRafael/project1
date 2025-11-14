import pandas as pd
import psycopg2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ruptures as rpt
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet

# --- CONFIGURACIÓN DE CONEXIÓN ---
# ¡ADVERTENCIA! Reemplaza estos valores con tus credenciales de PostgreSQL.
DB_HOST = "localhost"
DB_NAME = "project1"
DB_USER = "postgres"
DB_PASS = "12345"
DB_PORT = 5432
# ---------------------------------

conn = None # Inicializar conn

try:
    # 1. Conexión usando los parámetros directos (Método recomendado para psycopg2)
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        port=DB_PORT
    )

    # 2. Abrir un cursor para ejecutar comandos
    cursor = conn.cursor()

    # 3. Testear la conexión
    cursor.execute("SELECT 1")
    # Si la ejecución es exitosa, la conexión funciona
    print("✅ Conexión a la base de datos exitosa!")

except psycopg2.Error as e:
    print(f"❌ Error al conectar a PostgreSQL: {e}")
    conn = None

# 4. Asegurarse de cerrar la conexión si se estableció
# if conn:
#     conn.close()
    # print("Conexión cerrada.")

sql_hourly = f"""
    SELECT * FROM datos.hourly;
    """
sql_daily = f"""
    SELECT * FROM datos.daily;
    """
sql_uv = f"""
    SELECT * FROM datos.uv;
    """

dfh = pd.read_sql(sql_hourly, conn)
dfd = pd.read_sql(sql_daily, conn)
dfuv = pd.read_sql(sql_uv, conn)

# --- Asumiendo que tu DataFrame de datos diarios se llama 'dfd' ---
# dfd = tu DataFrame cargado de la tabla daily

# 1. Eliminar las primeras 11 filas
print(f"Filas iniciales: {len(dfd)}")
dfd = dfd.iloc[11:].copy()
print(f"Filas después de la limpieza inicial: {len(dfd)}")

# 2. Asegurar que el índice sea de tipo datetime (fundamental para series de tiempo)
# Asumimos que la columna 'time' (o el índice) ya ha sido convertida a DATE en la carga.
if 'time' in dfd.columns:
    dfd['time'] = pd.to_datetime(dfd['time'])
    dfd = dfd.set_index('time')
    
# 3. Asegurar que la variable principal sea numérica
variable_estudio = 'precipitation_probability_mean'
dfd[variable_estudio] = pd.to_numeric(dfd[variable_estudio], errors='coerce')

# 4. Eliminar filas donde la variable de estudio principal sea NaN
dfd = dfd.dropna(subset=[variable_estudio])

# Estadísticas descriptivas de la variable de estudio
print("\n--- Estadísticas Descriptivas ---")
print(dfd[variable_estudio].describe())

# Histograma y Boxplot
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
sns.histplot(dfd[variable_estudio], kde=True)
plt.title(f'Distribución de {variable_estudio}')

plt.subplot(1, 2, 2)
sns.boxplot(y=dfd[variable_estudio])
plt.title(f'Boxplot de {variable_estudio}')
plt.show()

# Gráfico de series de tiempo para identificar tendencia y estacionalidad visualmente
plt.figure(figsize=(14, 6))
dfd[variable_estudio].plot(title=f'Serie de Tiempo: {variable_estudio}')
plt.xlabel('Fecha')
plt.ylabel('Probabilidad (%)')
plt.grid(True)
plt.show()

# La estacionalidad se puede ajustar (ej. 365.25 días para estacionalidad anual)
# Si tus datos son muy largos, puedes probar 'period=365' o 'period=365.25'
decomposition = seasonal_decompose(dfd[variable_estudio], model='additive', period=7) # Asumiendo estacionalidad anual

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
decomposition.observed.plot(ax=ax1, title='Observado')
decomposition.trend.plot(ax=ax2, title='Tendencia')
decomposition.seasonal.plot(ax=ax3, title='Estacionalidad')
decomposition.resid.plot(ax=ax4, title='Residuos (Anomalías)')
plt.tight_layout()
plt.show()

# Calcular la matriz de correlación solo para variables numéricas
corr_matrix = dfd.corr()

# Mostrar la correlación de la variable de estudio con todas las demás
print(f"\n--- Correlación de {variable_estudio} con otras variables ---")
print(corr_matrix[variable_estudio].sort_values(ascending=False).head(10))

# Visualizar la matriz de correlación (heatmap)
plt.figure(figsize=(16, 14))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title('Matriz de Correlación de Variables Diarias')
plt.show()

# 1. Preparar DataFrame para Prophet
df_prophet = dfd[[variable_estudio]].reset_index()
df_prophet.rename(columns={'time': 'ds', variable_estudio: 'y'}, inplace=True)
df_prophet['ds'] = df_prophet['ds'].dt.date # Prophet es más feliz con solo DATE para datos diarios

print("\nDataFrame listo para Prophet:")
print(df_prophet.head())


# 1. Inicializar y entrenar el modelo
model = Prophet(
    # Quita las lineas que importaste antes y stan_backend
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

# 1. Identificar Anomalías
# Un día es anómalo si el valor real está por encima o por debajo del intervalo
df_anomalia['anomaly'] = 0
df_anomalia.loc[df_anomalia['y'] > df_anomalia['yhat_upper'], 'anomaly'] = 1  # Anomalía alta
df_anomalia.loc[df_anomalia['y'] < df_anomalia['yhat_lower'], 'anomaly'] = -1 # Anomalía baja

# 2. Filtrar y mostrar los días anómalos
dias_anomalos = df_anomalia[df_anomalia['anomaly'] != 0].copy()

print("\n--- Días Anómalos Detectados ---")
print(dias_anomalos[['y', 'yhat', 'anomaly']].sort_index())
print(f"\nTotal de días anómalos encontrados: {len(dias_anomalos)}")

# 1. Gráfico de la serie de tiempo y la predicción
plt.figure(figsize=(16, 7))
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
plt.show()

# Prophet también tiene una función de visualización incorporada
# model.plot(forecast)
# model.plot_components(forecast)



