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
def save_model_to_gcs(model_to_save, bucket_name, destination_blob):
    """Guarda el modelo en GCS mediante streaming de archivos para soportar pesos altos."""
    try:
        # Instanciar el cliente nativo de Google Cloud Storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob)
        
        # 💡 SOLUCIÓN REAL: Abrir un canal de streaming binario directo con el Blob de GCS
        # Esto fragmenta automáticamente el modelo pesado durante la transferencia
        with blob.open("wb", content_type="application/octet-stream") as f:
            pickle.dump(model_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        print(f"✅ Checkpoint pesado guardado con éxito vía Streaming en: gs://{bucket_name}/{destination_blob}")
        return True
    except Exception as e:
        # Este mensaje detallado aparecerá en la consola de Logs de Cloud Run para auditoría
        print(f"❌ Error crítico en streaming hacia GCS: {e}")
        return False

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

# --- BOTÓN DE GUARDADO MANUAL EN LA BARRA LATERAL (CORREGIDO) ---
st.sidebar.markdown("---")
st.sidebar.subheader("Persistencia del Modelo")
if st.sidebar.button("💾 Guardar Checkpoint en GCS"):
    # 💡 CORRECCIÓN: Usamos st.spinner() de forma global para que funcione nativamente
    with st.spinner("Subiendo modelo pesado a GCS..."):
        # Extraemos el objeto de la sesión para evitar referencias cruzadas
        raw_model = st.session_state.model
        
        # Ejecutamos la función de persistencia limpia
        success = save_model_to_gcs(raw_model, bucket_name, MODEL_PATH)
        
        if success:
            st.sidebar.success("¡Checkpoint guardado exitosamente en GCS!")
        else:
            st.sidebar.error("Error al guardar en GCS. Revisa los logs del contenedor.")

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

    # Una sola estructura limpia para almacenar el histórico de tuplas
    st.session_state.history = []
    st.session_state.history_file = []
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
    st.session_state.history = []
    st.session_state.history_file = []
    st.session_state.processed_files = []
    st.session_state.blobs = None
    st.session_state.index = 0
    st.rerun()

model = st.session_state.model
metric_acc = st.session_state.metric_acc
metric_prec = st.session_state.metric_prec
metric_rec = st.session_state.metric_rec

# =========================================================
# PREPROCESAMIENTO DE CHUNKS
# =========================================================
def preprocess_chunk(df_chunk):
    """Aplica todos los pasos de preprocesamiento a un fragmento de DataFrame."""
    # Clonamos para evitar advertencias de copia en Pandas
    df_chunk = df_chunk.copy()

    # 1. Asegurar conversión estricta a numérico de las coordenadas
    df_chunk['latitude'] = pd.to_numeric(df_chunk['latitude'], errors='coerce')
    df_chunk['longitude'] = pd.to_numeric(df_chunk['longitude'], errors='coerce')

    # 2. Quitar todos los nulos (incluyendo los generados por errores de conversión)
    df_chunk = df_chunk.dropna(subset=['latitude', 'longitude', 'domestic', 'district']).copy()

    # 3. Filtrar los registros con base en la latitud y longitud permitida para Chicago
    min_latitude, max_latitude = 41.644, 42.023
    min_longitude, max_longitude = -87.940, -87.524

    df_chunk = df_chunk[
        (df_chunk['latitude'] >= min_latitude) & (df_chunk['latitude'] <= max_latitude) &
        (df_chunk['longitude'] >= min_longitude) & (df_chunk['longitude'] <= max_longitude)
    ].copy()

    # Convertir la columna 'date' a datetime
    df_chunk['date'] = pd.to_datetime(df_chunk['date'], utc=True, errors='coerce')
    df_chunk = df_chunk.dropna(subset=['date']).copy()

    # 4. Generar columnas de tiempo
    df_chunk['day_of_week'] = df_chunk['date'].dt.day_name()
    df_chunk['month'] = df_chunk['date'].dt.month_name()
    df_chunk['hour_of_day'] = df_chunk['date'].dt.hour.astype(float)
    df_chunk['is_weekend'] = df_chunk['day_of_week'].isin(['Saturday', 'Sunday'])

    # Convertir 'district' a string de forma homogénea
    df_chunk['district'] = df_chunk['district'].astype(str)

    # Columnas finales requeridas por el ARFClassifier
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

    file_acc = metrics.Accuracy()
    file_prec = metrics.Precision()
    file_rec = metrics.Recall()

    try:
        content = blob.download_as_bytes()
        buffer = io.BytesIO(content)

        # 💡 SOLUCIÓN: Leemos latitude y longitude como strings inicialmente para evitar el quiebre de tipos
        for chunk in pd.read_csv(buffer, chunksize=chunksize, low_memory=False, 
                                 dtype={'latitude': str, 'longitude': str, 'district': str}):
            
            if not expected_raw_cols.issubset(set(chunk.columns)):
                continue

            # El preprocesamiento ahora se encarga de limpiar y convertir a números de forma segura
            X_chunk, y_chunk = preprocess_chunk(chunk)

            if X_chunk.empty:
                continue

            for idx, row in X_chunk.iterrows():
                if count >= limite:
                    break

                try:
                    x = row.to_dict()
                    x['domestic'] = bool(x['domestic'])
                    x['is_weekend'] = bool(x['is_weekend'])
                    
                    y = bool(y_chunk.loc[idx])

                    # Predict-then-Learn
                    y_pred = model.predict_one(x)
                    if y_pred is None:
                        y_pred = False 

                    model.learn_one(x, y)

                    # Actualizar métricas globales
                    metric_acc.update(y, y_pred)
                    metric_prec.update(y, y_pred)
                    metric_rec.update(y, y_pred)

                    # Actualizar métricas del archivo
                    file_acc.update(y, y_pred)
                    file_prec.update(y, y_pred)
                    file_rec.update(y, y_pred)

                    count += 1
                except Exception:
                    continue

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
# CONTROLADOR DE PROCESAMIENTO MIGRADO (Piloto Automático Seguro)
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

# Inicializar la variable de control para el Piloto Automático
if "auto_pilot_remaining" not in st.session_state:
    st.session_state.auto_pilot_remaining = 0

blobs = st.session_state.blobs
idx = st.session_state.index

# Creamos dos columnas para organizar visualmente los botones
btn_col1, btn_col2 = st.columns(2)

# --- OPCIÓN 1: PROCESAR UN SOLO ARCHIVO ---
with btn_col1:
    # El botón se deshabilita automáticamente si el piloto automático está corriendo
    disabled_btn = st.session_state.auto_pilot_remaining > 0
    if st.button("Procesar Siguiente Archivo ➡️", disabled=disabled_btn):
        if idx >= len(blobs):
            st.success("¡Todos los lotes de crímenes han sido procesados!")
        else:
            blob = blobs[idx]
            short_name = blob.name.split("/")[-1]
            st.info(f"Procesando lote {idx + 1}/{len(blobs)}: `{short_name}`")
            
            with st.spinner("Entrenando ARFClassifier..."):
                result = process_single_blob(bucket_name, blob.name, int(limite), int(chunksize))
            
            if result is not None:
                st.session_state.history.append((result["global_acc"], result["global_prec"], result["global_rec"]))
                st.session_state.history_file.append((result["file_acc"], result["file_prec"], result["file_rec"]))
                st.session_state.processed_files.append(short_name)
            
            st.session_state.index += 1
            st.rerun()

# --- OPCIÓN 2: PILOTO AUTOMÁTICO RE-DISEÑADO (Evita Timeouts) ---
with btn_col2:
    if st.session_state.auto_pilot_remaining == 0:
        num_bloque = st.number_input("Archivos a procesar en ráfaga:", value=50, min_value=1, max_value=100, step=10)
        if st.button("🚀 Iniciar Piloto Automático"):
            archivos_restantes = len(blobs) - idx
            st.session_state.auto_pilot_remaining = min(int(num_bloque), archivos_restantes)
            st.rerun()
    else:
        # Permite al usuario frenar la ráfaga de forma segura si lo desea
        if st.button("🛑 Detener Piloto Automático"):
            st.session_state.auto_pilot_remaining = 0
            st.warning("Piloto automático detenido por el usuario.")
            st.rerun()

# --- EJECUCIÓN DEL PILOTO AUTOMÁTICO (Ciclo continuo por recarga de página) ---
if st.session_state.auto_pilot_remaining > 0:
    if idx >= len(blobs):
        st.session_state.auto_pilot_remaining = 0
        st.success("¡Todos los archivos del bucket han sido procesados por completo!")
    else:
        blob = blobs[idx]
        short_name = blob.name.split("/")[-1]
        
        # Banner informativo sobre el estado de la ráfaga automática
        st.warning(f"🤖 **Modo Piloto Automático Activo.** Quedan **{st.session_state.auto_pilot_remaining}** archivos en esta ráfaga.")
        st.info(f"Procesando dinámicamente: `{short_name}` (Índice global: {idx + 1})")
        
        # Procesamos el archivo actual
        result = process_single_blob(bucket_name, blob.name, int(limite), int(chunksize))
        
        if result is not None:
            st.session_state.history.append((result["global_acc"], result["global_prec"], result["global_rec"]))
            st.session_state.history_file.append((result["file_acc"], result["file_prec"], result["file_rec"]))
            st.session_state.processed_files.append(short_name)
        
        # Actualizamos contadores de control
        st.session_state.auto_pilot_remaining -= 1
        st.session_state.index += 1
        
        # Recarga inmediata: refresca la UI, dibuja un nuevo punto en los gráficos y pasa al siguiente CSV
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
if st.session_state.history and len(st.session_state.processed_files) > 0:
    
    # Reconstruimos las listas individuales para el DataFrame a partir de las tuplas guardadas
    accuracies_global = [h[0] for h in st.session_state.history]
    precisions_global = [h[1] for h in st.session_state.history]
    recalls_global = [h[2] for h in st.session_state.history]

    accuracies_local = [h[0] for h in st.session_state.history_file]
    precisions_local = [h[1] for h in st.session_state.history_file]
    recalls_local = [h[2] for h in st.session_state.history_file]

    # Asegurar alineación de índices por seguridad en la UI
    min_len = min(len(st.session_state.processed_files), len(st.session_state.history))
    
    df_metrics = pd.DataFrame({
        "Lote/Archivo": st.session_state.processed_files[:min_len],
        "Accuracy (Global)": accuracies_global[:min_len],
        "Precision (Global)": precisions_global[:min_len],
        "Recall (Global)": recalls_global[:min_len],
        "Accuracy (Este Lote)": accuracies_local[:min_len],
        "Precision (Este Lote)": precisions_local[:min_len],
        "Recall (Este Lote)": recalls_local[:min_len],
    })

    st.markdown("---")
    st.subheader("📈 Monitoreo de Aprendizaje Continuo")

    tab1, tab2, tab3 = st.tabs([
        "📊 Métricas Acumuladas (Histórico)", 
        "⚡ Desempeño por Lote Aislado", 
        "📋 Tabla de Datos Real"
    ])
    
    with tab1:
        st.markdown("#### Evolución de Métricas Globales Acumuladas")
        st.line_chart(
            df_metrics.set_index("Lote/Archivo")[["Accuracy (Global)", "Precision (Global)", "Recall (Global)"]],
            color=["#2ca02c", "#1f77b4", "#ff7f0e"]
        )
        
    with tab2:
        st.markdown("#### Comportamiento del Modelo en el Lote Actual")
        st.line_chart(
            df_metrics.set_index("Lote/Archivo")[["Accuracy (Este Lote)", "Precision (Este Lote)", "Recall (Este Lote)"]],
            color=["#4ade80", "#60a5fa", "#f87171"]
        )
        
    with tab3:
        st.dataframe(df_metrics, use_container_width=True)

else:
    st.markdown("---")
    st.info("💡 Las gráficas de evolución aparecerán aquí en tiempo real en cuanto proceses el primer archivo CSV.")

st.caption("Ecosistema: Cloud Run • River ML • Chicago Crime Dataset Abierto")
