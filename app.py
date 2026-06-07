import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle
from google.cloud import storage
import river
from river import forest, metrics

# =========================================================
# CONFIGURACIÓN DE PÁGINA
# =========================================================
st.set_page_config(page_title="Predicción de Arrestos - Chicago", page_icon="👮🏽‍♂️", layout="wide")
st.title("Aprendizaje en Línea: Crímenes de Chicago 👮🏽‍♂️")

st.markdown("""
Este panel interactivo entrena un modelo **Adaptive Random Forest Classifier** de forma incremental,
procesando los datos históricos de crímenes de Chicago **un archivo por clic** desde Google Cloud Storage (GCS).

**Enfoque Online:** Por cada registro, el modelo primero predice si habrá un arresto (`Predict`) y luego aprende de la realidad (`Learn`).

Los datos provienen del sistema CLEAR (Citizen Law Enforcement Analysis and Reporting) del Departamento de Policía de Chicago (CPD). 
Se encuentran alojados de forma pública en el Portal de Datos Abiertos de la Ciudad de Chicago (Chicago Data Portal) y está 
disponible para su consulta masiva mediante la base de datos pública de Google BigQuery bajo el 
identificador bigquery-public-data.chicago_crime.crime.

""")

# =========================================================
# FUNCIONES AUXILIARES GCS
# =========================================================
def save_model_to_gcs(model, bucket_name, destination_blob):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob)
        blob.upload_from_string(pickle.dumps(model))
        st.success(f"Checkpoint guardado en GCS: `{destination_blob}`")
    except Exception as e:
        st.warning(f"No se pudo respaldar el modelo en GCS: {e}")

def load_model_from_gcs(bucket_name, source_blob):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob)

        if blob.exists():
            data = blob.download_as_bytes()
            st.info("Modelo incremental recuperado desde GCS.")
            return pickle.loads(data)
        return None
    except Exception as e:
        st.warning(f"No se pudo cargar el modelo previo: {e}")
        return None

def delete_model_from_gcs(bucket_name, source_blob):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob)
        if blob.exists():
            blob.delete()
            st.success("Modelo eliminado de GCS.")
        else:
            st.info("No se encontró ningún modelo guardado en GCS.")
    except Exception as e:
        st.warning(f"Error al eliminar el modelo: {e}")

# =========================================================
# INICIALIZACIÓN DEL MODELO NUEVO
# =========================================================
def new_model():
    """
    Instancia el clasificador forestal adaptativo del notebook.
    """
    return forest.ARFClassifier(n_models=10, seed=42)

# =========================================================
# PARÁMETROS DE LA INTERFAZ
# =========================================================
st.sidebar.header("Configuración de Datos")
bucket_name = st.sidebar.text_input("Bucket de GCS:", "bucket-ml-bd-gusmercado")
prefix = st.sidebar.text_input("Prefijo/Carpeta:", "chicago_crime/")
limite = st.sidebar.number_input("Muestras por archivo (Límite):", value=5000, step=500)
chunksize = st.sidebar.number_input("Tamaño del Chunk de lectura:", value=5000, step=500) 

MODEL_PATH = "models/model_incremental.pkl"

# BOTÓN DE GUARDADO MANUAL EN LA BARRA LATERAL
st.sidebar.markdown("---")
st.sidebar.subheader("Persistencia del Modelo")
if st.sidebar.button("💾 Guardar Checkpoint en GCS"):
    with st.sidebar.spinner("Subiendo modelo pesado a GCS..."):
        save_model_to_gcs(st.session_state.model, bucket_name, MODEL_PATH)
# -----------------------------------------------------------------------

# =========================================================
# INICIALIZAR SESSION STATE (Métricas de Clasificación)
# =========================================================
if "model" not in st.session_state:
    try:
        loaded_model = load_model_from_gcs(bucket_name, MODEL_PATH)
    except Exception:
        loaded_model = None

    if loaded_model is None:
        loaded_model = new_model()

    st.session_state.model = loaded_model
    st.session_state.metric_acc = metrics.Accuracy()
    st.session_state.metric_prec = metrics.Precision()
    st.session_state.metric_rec = metrics.Recall()

    st.session_state.history_acc = []
    st.session_state.history_prec = []
    st.session_state.history_rec = []
    
    st.session_state.history_file_acc = []
    st.session_state.history_file_prec = []
    st.session_state.history_file_rec = []
    
    st.session_state.processed_files = []
    st.session_state.blobs = None
    st.session_state.index = 0

# Botón de reinicio completo
if st.sidebar.button("Reiniciar Modelo y Borrar Historial"):
    delete_model_from_gcs(bucket_name, MODEL_PATH)
    st.session_state.model = new_model()
    st.session_state.metric_acc = metrics.Accuracy()
    st.session_state.metric_prec = metrics.Precision()
    st.session_state.metric_rec = metrics.Recall()
    st.session_state.history_acc = []
    st.session_state.history_prec = []
    st.session_state.history_rec = []
    st.session_state.history_file_acc = []
    st.session_state.history_file_prec = []
    st.session_state.history_file_rec = []
    st.session_state.processed_files = []
    st.session_state.blobs = None
    st.session_state.index = 0
    st.rerun()

# Variables globales de la sesión para simplificar llamadas
model = st.session_state.model
metric_acc = st.session_state.metric_acc
metric_prec = st.session_state.metric_prec
metric_rec = st.session_state.metric_rec

# =========================================================
# PREPROCESAMIENTO DE CHUNKS (Adaptado del Notebook)
# =========================================================
def preprocess_chunk(df_chunk):
    df_chunk = df_chunk.dropna(subset=['latitude', 'longitude', 'domestic', 'district']).copy()
    
    # Límites geográficos de Chicago
    df_chunk = df_chunk[
        df_chunk['latitude'].between(41.644, 42.023) &
        df_chunk['longitude'].between(-87.940, -87.524)
    ].copy()

    df_chunk['date'] = pd.to_datetime(df_chunk['date'], utc=True, errors='coerce')
    df_chunk = df_chunk.dropna(subset=['date']).copy()

    # Extracción de variables de tiempo
    df_chunk['day_of_week'] = df_chunk['date'].dt.day_name()
    df_chunk['month'] = df_chunk['date'].dt.month_name()
    df_chunk['hour_of_day'] = df_chunk['date'].dt.hour.astype(float)
    df_chunk['is_weekend'] = df_chunk['day_of_week'].isin(['Saturday', 'Sunday'])
    df_chunk['district'] = df_chunk['district'].astype(str)

    # Columnas finales requeridas para entrenar
    features = [
        'primary_type', 'location_description', 'domestic', 'district', 
        'latitude', 'longitude', 'day_of_week', 'month', 'hour_of_day', 'is_weekend'
    ]
    
    return df_chunk[features], df_chunk['arrest']

# =========================================================
# PIPELINE DE ENTRENAMIENTO ONLINE POR ARCHIVO
# =========================================================
def process_single_blob(bucket_name, blob_name, limite, chunksize):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    expected_raw_cols = {'date', 'primary_type', 'location_description', 'arrest', 
                         'domestic', 'district', 'latitude', 'longitude'}
    count = 0

    # Métricas exclusivas para este archivo
    file_acc = metrics.Accuracy()
    file_prec = metrics.Precision()
    file_rec = metrics.Recall()

    try:
        content = blob.download_as_bytes()
        buffer = io.BytesIO(content)

        for chunk in pd.read_csv(buffer, chunksize=chunksize, low_memory=False):
            if not expected_raw_cols.issubset(set(chunk.columns)):
                continue

            X_chunk, y_chunk = preprocess_chunk(chunk)

            for idx, row in X_chunk.iterrows():
                if count >= limite:
                    break

                # Formateo estricto de tipos nativos para River
                x = row.to_dict()
                x['domestic'] = bool(x['domestic'])
                x['is_weekend'] = bool(x['is_weekend'])
                y = bool(y_chunk.loc[idx])

                # 1. Predict-then-Learn
                y_pred = model.predict_one(x)
                
                # River inicializa en None la primera predicción
                if y_pred is None:
                    y_pred = False 

                model.learn_one(x, y)

                # 2. Actualizar métricas acumuladas globales
                metric_acc.update(y, y_pred)
                metric_prec.update(y, y_pred)
                metric_rec.update(y, y_pred)

                # 3. Actualizar métricas locales del archivo
                file_acc.update(y, y_pred)
                file_prec.update(y, y_pred)
                file_rec.update(y, y_pred)

                count += 1

            if count >= limite:
                break

        if count == 0:
            return None

        return {
            "count": count,
            "file_acc": file_acc.get(),
            "file_prec": file_prec.get(),
            "file_rec": file_rec.get(),
            "global_acc": metric_acc.get(),
            "global_prec": metric_prec.get(),
            "global_rec": metric_rec.get()
        }

    except Exception as e:
        st.error(f"Error procesando `{blob_name}`: {e}")
        return None

# =========================================================
# CONTROLADOR DE PROCESAMIENTO MIGRADO (Individual y Bloques)
# =========================================================
st.subheader("Flujo de Simulación en Tiempo Real")

# Inicializar los blobs si no existen en la sesión
if st.session_state.blobs is None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    blobs = [b for b in blobs if b.name.lower().endswith(".csv")]
    st.session_state.blobs = blobs
    st.session_state.index = 0

blobs = st.session_state.blobs
idx = st.session_state.index

# Creamos dos columnas para colocar los botones lado a lado
btn_col1, btn_col2 = st.columns(2)

# --- OPCIÓN 1: PROCESAR UN SOLO ARCHIVO ---
with btn_col1:
    if st.button("Procesar Siguiente Archivo ➡️"):
        if idx >= len(blobs):
            st.success("¡Todos los lotes de crímenes han sido procesados con éxito!")
        else:
            blob = blobs[idx]
            short_name = blob.name.split("/")[-1]
            st.info(f"Procesando lote {idx + 1}/{len(blobs)}: `{short_name}`")
            
            with st.spinner("Entrenando..."):
                result = process_single_blob(bucket_name, blob.name, int(limite), int(chunksize))
            
            if result is not None:
                st.session_state.history_acc.append(result["global_acc"])
                st.session_state.history_prec.append(result["global_prec"])
                st.session_state.history_rec.append(result["global_rec"])
                st.session_state.history_file_acc.append(result["file_acc"])
                st.session_state.history_file_prec.append(result["file_prec"])
                st.session_state.history_file_rec.append(result["file_rec"])
                st.session_state.processed_files.append(short_name)
                st.success("¡Lote procesado en memoria!")
            
            st.session_state.index += 1
            st.rerun()

# --- OPCIÓN 2: PROCESAR EN BLOQUE (Ej. 50 archivos) ---
with btn_col2:
    num_bloque = st.number_input("Archivos a procesar en ráfaga:", value=50, min_value=1, max_value=100, step=10)
    if st.button("🚀 Procesar Bloque en Ráfaga"):
        if idx >= len(blobs):
            st.success("¡Todos los lotes ya han sido procesados!")
        else:
            # Determinamos cuántos archivos procesar sin pasarnos del límite del array
            archivos_restantes = len(blobs) - idx
            iteraciones = min(int(num_bloque), archivos_restantes)
            
            # Contenedor visual para el progreso de la ráfaga
            progreso_bar = st.progress(0.0)
            status_text = st.empty()
            
            for i in range(iteraciones):
                actual_idx = st.session_state.index
                blob = blobs[actual_idx]
                short_name = blob.name.split("/")[-1]
                
                status_text.markdown(f"⚡ Ráfaga: Procesando {i+1}/{iteraciones} (`{short_name}`)...")
                
                # Ejecutamos el entrenamiento del archivo actual
                result = process_single_blob(bucket_name, blob.name, int(limite), int(chunksize))
                
                if result is not None:
                    st.session_state.history_acc.append(result["global_acc"])
                    st.session_state.history_prec.append(result["global_prec"])
                    st.session_state.history_rec.append(result["global_rec"])
                    st.session_state.history_file_acc.append(result["file_acc"])
                    st.session_state.history_file_prec.append(result["file_prec"])
                    st.session_state.history_file_rec.append(result["file_rec"])
                    st.session_state.processed_files.append(short_name)
                
                # Avanzamos el índice global
                st.session_state.index += 1
                # Actualizamos la barra de progreso de la UI
                progreso_bar.progress((i + 1) / iteraciones)
            
            status_text.empty()
            progreso_bar.empty()
            st.success(f"¡Ráfaga completada! Se procesaron {iteraciones} archivos seguidos en la RAM.")
            st.rerun()

# =========================================================
# INDICADORES EN PANTALLA (KPIs)
# =========================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Lote Actual", f"{st.session_state.index}")
col2.metric("Accuracy Acumulada", f"{metric_acc.get():.4f}")
col3.metric("Precision Acumulada", f"{metric_prec.get():.4f}")
col4.metric("Recall Acumulado", f"{metric_rec.get():.4f}")

# =========================================================
# SECCIÓN GRÁFICA DEL HISTORIAL (Evolución en el Tiempo)
# =========================================================
if st.session_state.history_acc:
    # 1. Creamos un DataFrame con todo el histórico acumulado en la sesión
    df_metrics = pd.DataFrame({
        "Lote/Archivo": st.session_state.processed_files,
        "Accuracy (Global)": st.session_state.history_acc,
        "Precision (Global)": st.session_state.history_prec,
        "Recall (Global)": st.session_state.history_rec,
        "Accuracy (Este Lote)": st.session_state.history_file_acc,
        "Precision (Este Lote)": st.session_state.history_file_prec,
        "Recall (Este Lote)": st.session_state.history_file_rec,
    })

    st.markdown("---")
    st.subheader("📈 Monitoreo de Aprendizaje Continuo")
    st.markdown("""
    Estas gráficas muestran el comportamiento del modelo a lo largo del tiempo. 
    El eje **X** representa el orden secuencial en el que los archivos CSV fueron leídos de GCS.
    """)

    # 2. Organizamos los gráficos en pestañas para una interfaz más limpia
    tab1, tab2, tab3 = st.tabs([
        "📊 Métricas Acumuladas (Histórico)", 
        "⚡ Desempeño por Lote Aislado", 
        "📋 Tabla de Datos Real"
    ])
    
    with tab1:
        st.markdown("#### Evolución de Métricas Globales Acumuladas")
        st.caption("Muestra la estabilidad general del modelo desde el inicio del entrenamiento hasta el momento actual.")
        # Graficamos las métricas globales indexando por el nombre del archivo
        st.line_chart(
            df_metrics.set_index("Lote/Archivo")[["Accuracy (Global)", "Precision (Global)", "Recall (Global)"]],
            color=["#2ca02c", "#1f77b4", "#ff7f0e"] # Verde, Azul, Naranja
        )
        
    with tab2:
        st.markdown("#### Comportamiento del Modelo en el Lote Actual")
        st.caption("Ideal para detectar variaciones drásticas o 'Concept Drift' en archivos específicos.")
        st.line_chart(
            df_metrics.set_index("Lote/Archivo")[["Accuracy (Este Lote)", "Precision (Este Lote)", "Recall (Este Lote)"]],
            color=["#4ade80", "#60a5fa", "#f87171"] # Versiones más brillantes para diferenciar
        )
        
    with tab3:
        st.markdown("#### Historial Registrado")
        st.dataframe(df_metrics, use_container_width=True)

else:
    st.markdown("---")
    st.info("💡 Las gráficas de evolución aparecerán aquí en tiempo real en cuanto proceses el primer archivo CSV.")

st.caption("Ecosistema: Cloud Run • River ML • Chicago Crime Dataset Abierto")
