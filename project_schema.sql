CREATE SCHEMA datos
  -- -----------------------------------------------------------
  -- TABLA HOURLY (Datos por hora)
  -- -----------------------------------------------------------
  CREATE TABLE IF NOT EXISTS hourly (
    time TIMESTAMP, 
    temperature_2m REAL,
    relative_humidity_2m REAL,
    dew_point_2m REAL,
    apparent_temperature REAL,
    precipitation_probability REAL,
    precipitation REAL,
    rain REAL,
    showers REAL,
    weather_code REAL,
    cloud_cover REAL,
    cloud_cover_low REAL,
    cloud_cover_mid REAL,
    cloud_cover_high REAL,
    et0_fao_evapotranspiration REAL,
    wind_speed_10m REAL,
    wind_speed_80m REAL,
    wind_speed_120m REAL,
    wind_speed_180m REAL,
    temperature_80m REAL,
    temperature_120m REAL,
    temperature_180m REAL,
    soil_temperature_0cm REAL,
    soil_temperature_6cm REAL,
    soil_temperature_18cm REAL,
    soil_temperature_54cm REAL,
    soil_moisture_0_to_1cm REAL,
    soil_moisture_1_to_3cm REAL,
    soil_moisture_3_to_9cm REAL,
    soil_moisture_9_to_27cm REAL,
    soil_moisture_27_to_81cm REAL
  )

  -- -----------------------------------------------------------
  -- TABLA DAILY (Resúmenes diarios)
  -- -----------------------------------------------------------
  CREATE TABLE IF NOT EXISTS daily (
    -- La columna 'time' es el identificador principal (día)
    time TIMESTAMP, 
    temperature_2m_max REAL,
    temperature_2m_min REAL,
    apparent_temperature_max REAL,
    apparent_temperature_min REAL,
    weather_code REAL,
    temperature_2m_mean REAL,
    relative_humidity_2m_max REAL,
    relative_humidity_2m_min REAL,
    relative_humidity_2m_mean REAL,
    precipitation_sum REAL,
    precipitation_probability_max REAL,
    rain_sum REAL,
    apparent_temperature_mean REAL,
    wind_speed_10m_max REAL,
    wind_gusts_10m_max REAL,
    cloud_cover_min REAL,
    cloud_cover_max REAL,
    cloud_cover_mean REAL,
    dew_point_2m_mean REAL,
    dew_point_2m_max REAL,
    dew_point_2m_min REAL,
    wind_gusts_10m_mean REAL,
    wind_speed_10m_mean REAL,
    wind_gusts_10m_min REAL,
    wind_speed_10m_min REAL,
    wet_bulb_temperature_2m_mean REAL,
    wet_bulb_temperature_2m_max REAL,
    wet_bulb_temperature_2m_min REAL,
    precipitation_probability_mean REAL,
    precipitation_probability_min REAL,
    sunrise TIMESTAMP,
    sunset TIMESTAMP,
    daylight_duration REAL,
    sunshine_duration REAL,
    showers_sum REAL,
    et0_fao_evapotranspiration REAL,
    et0_fao_evapotranspiration_sum REAL
  )

  -- -----------------------------------------------------------
  -- TABLA UV (Datos relacionados con UV y Radiación)
  -- -----------------------------------------------------------
  CREATE TABLE IF NOT EXISTS uv (
    -- La columna 'time' como clave, suponiendo que es un resumen diario
    time TIMESTAMP, 
    weather_code REAL,
    daylight_duration REAL,
    cloud_cover_mean REAL,
    cloud_cover_max REAL,
    cloud_cover_min REAL,
    et0_fao_evapotranspiration REAL,
    shortwave_radiation_sum REAL,
    uv_index_max REAL,
    uv_index_clear_sky_max REAL
  );
  
SELECT 'Copying data into datos.hourly';
\copy datos.hourly FROM 'C:/Users/je_z_/OneDrive/Documentos/bot_metin_2/Tarea/Project/data/datosmeteorogicoshoras.csv' DELIMITER E',' CSV HEADER;
SELECT 'Copying data into datos.daily';
\copy datos.daily FROM 'C:/Users/je_z_/OneDrive/Documentos/bot_metin_2/Tarea/Project/data/datosmeteorologicosdiarios.csv' DELIMITER E',' CSV HEADER; 
SELECT 'Copying data into datos.uv';
\copy datos.uv FROM 'C:/Users/je_z_/OneDrive/Documentos/bot_metin_2/Tarea/Project/data/datosuv.csv' DELIMITER E',' CSV HEADER;