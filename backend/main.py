from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os

# --- CONFIGURACIÓN INICIAL ---
app = FastAPI(
    title="API de Predicción de Riesgo (Callao)",
    description="Backend para predecir riesgo delictivo usando Gradient Boosting",
    version="1.0.0"
)

# Variables globales para los modelos
model_gb = None
encoder = None
cluster_finder = None

# --- CARGA DE MODELOS AL INICIO ---
@app.on_event("startup")
def load_models():
    global model_gb, encoder, cluster_finder
    try:
        # Rutas relativas a la carpeta 'models'
        base_dir = os.path.dirname(__file__)
        models_dir = os.path.join(base_dir, "models")
        
        print("Cargando artefactos de IA...")
        model_gb = joblib.load(os.path.join(models_dir, "modelo_riesgo_gb.pkl"))
        encoder = joblib.load(os.path.join(models_dir, "encoder_features.pkl"))
        cluster_finder = joblib.load(os.path.join(models_dir, "buscador_cluster.pkl"))
        print("¡Modelos cargados correctamente!")
    except Exception as e:
        print(f"ERROR CRÍTICO: No se pudieron cargar los modelos. {e}")
        # Esto detiene el servidor si faltan archivos
        raise e

# --- ESQUEMAS DE DATOS (Pydantic) ---
class InputDatos(BaseModel):
    latitud: float
    longitud: float
    fecha_hora: str  # Formato esperado: "YYYY-MM-DD HH:MM:SS"
    # Ejemplo: "2024-12-05 18:30:00"

class OutputPrediccion(BaseModel):
    nivel_riesgo: str
    probabilidad: float
    mensaje: str
    datos_procesados: dict

# --- LÓGICA DE PREDICCIÓN (UNO POR UNO) ---
def procesar_y_predecir(lat, lon, fecha_str):
    """
    Toma datos crudos del usuario y los convierte en la entrada que el modelo entiende.
    """
    # 1. Procesar Fecha y Hora
    try:
        dt = datetime.strptime(fecha_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        raise ValueError("Formato de fecha incorrecto. Use: YYYY-MM-DD HH:MM:SS")
    
    hora = dt.hour
    dia_semana = dt.weekday() # 0=Lunes, 6=Domingo

    # 2. Encontrar el Cluster (Zona)
    # El KNN espera [[lat, lon]]
    cluster_id = cluster_finder.predict([[lat, lon]])[0]

    # 3. Crear DataFrame de UNA sola fila
    # IMPORTANTE: Los nombres de columnas deben ser idénticos al entrenamiento
    input_df = pd.DataFrame({
        'CLUSTER_ID': [cluster_id],
        'DIA_SEMANA': [dia_semana],
        'HORA_DEL_DIA': [hora]
    })

    # 4. Codificar (Encoder)
    # Transformamos esa fila única a formato One-Hot
    input_encoded = encoder.transform(input_df)

    # 5. Predecir (Gradient Boosting)
    prediccion_clase = model_gb.predict(input_encoded)[0] # 'Alto' o 'Bajo'
    
    # Obtener probabilidad (confianza)
    probs = model_gb.predict_proba(input_encoded)[0]
    # La clase 1 suele ser la positiva/riesgo, pero verificamos con clases_
    clases = model_gb.classes_
    idx_clase = np.where(clases == prediccion_clase)[0][0]
    probabilidad = probs[idx_clase]

    return prediccion_clase, probabilidad, int(cluster_id), dia_semana, hora

# --- ENDPOINT (Ruta API) ---
@app.post("/predict", response_model=OutputPrediccion)
def predecir_riesgo(datos: InputDatos):
    try:
        # Llamamos a nuestra función lógica
        resultado, prob, zona, dia, hora = procesar_y_predecir(
            datos.latitud, 
            datos.longitud, 
            datos.fecha_hora
        )
        
        # Generamos un mensaje amigable
        if resultado == "Alto":
            msg = "PRECAUCIÓN: Se detecta alta probabilidad de incidencia delictiva en esta zona y hora."
        else:
            msg = "Zona con bajo riesgo histórico detectado."

        return {
            "nivel_riesgo": resultado,
            "probabilidad": round(prob, 4),
            "mensaje": msg,
            "datos_procesados": {
                "zona_id": zona,
                "dia_semana_idx": dia,
                "hora": hora
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINT DE SALUD (Health Check) ---
@app.get("/")
def home():
    return {"status": "online", "model": "Gradient Boosting"}