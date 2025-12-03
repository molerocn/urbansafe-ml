#!/usr/bin/env python
# coding: utf-8

# # Configuración y carga de datos
# 
# Import de librerías necesarias para el entorno de trabajo y carga del archivo csv con pandas.[link text](https://)

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from itertools import product

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.svm import SVC


# In[2]:


file_path = "https://raw.githubusercontent.com/molerocn/urbansafe-ml/main/data/raw/registros_delictivos.csv"
df = pd.read_csv(file_path, delimiter=';')
df.info()


# # Limpieza y filtrado de datos

# Verificar si existen datos nulos

# In[3]:


columnas_clave = ['LONGITUD', 'LATITUD', 'HORA_HECHO', 'FECHA_HECHO']
df[columnas_clave].isnull().sum()


# Eliminación de datos nulos

# In[4]:


df_general = df.dropna(subset=['LONGITUD', 'LATITUD']).copy()
df_general[columnas_clave].isnull().sum()


# # Ingeniería de características

# In[5]:


df_general['HORA_DEL_DIA'] = df_general['HORA_HECHO'] // 100

df_general['HORA_DEL_DIA'] = df_general['HORA_DEL_DIA'].apply(lambda x: 0 if x >= 24 else x)
df_general[['HORA_HECHO', 'HORA_DEL_DIA']].sample(5)


# In[6]:


df_general['FECHA_HECHO'] = pd.to_datetime(df_general['FECHA_HECHO'], format='%Y%m%d')

df_general['DIA_SEMANA'] = df_general['FECHA_HECHO'].dt.dayofweek
df_general[['FECHA_HECHO', 'DIA_SEMANA']].sample(5)


# # Análisis exploratorio

# In[7]:


plt.figure(figsize=(14, 7))
sns.countplot(
    data=df_general,
    x='HORA_DEL_DIA',
)
plt.title('Número de Incidentes (General) por Hora del Día', fontsize=16)
plt.xlabel('Hora del Día (0-23)', fontsize=12)
plt.ylabel('Cantidad de Incidentes', fontsize=12)
plt.xticks(range(0, 24))
plt.show()


# In[8]:


dias_map = {
    0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves',
    4: 'Viernes', 5: 'Sábado', 6: 'Domingo'
}
df_general['DIA_NOMBRE'] = df_general['DIA_SEMANA'].map(dias_map)

plt.figure(figsize=(12, 6))
sns.countplot(
    data=df_general,
    x='DIA_NOMBRE',
    order=dias_map.values(),
)
plt.title('Número de Incidentes (General) por Día de la Semana', fontsize=16)
plt.xlabel('Día de la Semana', fontsize=12)
plt.ylabel('Cantidad de Incidentes', fontsize=12)
plt.show()


# In[9]:


plt.figure(figsize=(10, 10))
sns.scatterplot(
    data=df_general,
    x='LONGITUD',
    y='LATITUD',
    s=10
)
plt.title('Mapa de Densidad de Incidentes (Riesgo General)', fontsize=16)
plt.xlabel('Longitud', fontsize=12)
plt.ylabel('Latitud', fontsize=12)
plt.show()


# # Clustering espacial

# ## Preparar y escalar datos de coordenadas

# In[10]:


coords = df_general[['LONGITUD', 'LATITUD']]
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)

print(coords_scaled[:5])


# ## Encontrar parametros (eps)

# In[11]:


min_samples_k = 10

nn = NearestNeighbors(n_neighbors=min_samples_k)
nn.fit(coords_scaled)

distances, indices = nn.kneighbors(coords_scaled)
k_distances = np.sort(distances[:, min_samples_k-1], axis=0)

plt.figure(figsize=(12, 7))
plt.plot(k_distances)
plt.title(f'Gráfico K-distance (para k={min_samples_k})', fontsize=16)
plt.xlabel('Puntos (ordenados por distancia)', fontsize=12)
plt.ylabel(f'Distancia al {min_samples_k}-ésimo vecino', fontsize=12)
plt.axhline(y=0.1, color='red', linestyle='--', label='Ej: Eps=0.1') # Línea de ejemplo
plt.axhline(y=0.2, color='green', linestyle='--', label='Ej: Eps=0.2') # Línea de ejemplo
plt.legend()
plt.show()

print("Observa el gráfico de arriba.")
print("Busca el punto donde la curva 'se dispara' (el codo).")
print("Ese valor en el eje Y (ej. 0.1, 0.15, 0.2) será tu EPSILON.")


# ## Ejecutar el clustering DBSCAN

# In[12]:


EPSILON_NUEVO = 0.07
MIN_SAMPLES = 10

dbscan = DBSCAN(eps=EPSILON_NUEVO, min_samples=MIN_SAMPLES)
clusters = dbscan.fit_predict(coords_scaled)
df_general['CLUSTER_ID'] = clusters

# aqui hice muchas pruebas profe, pero me quede con eps 0.07
print(f"Nuevo Clustering con eps={EPSILON_NUEVO} ---")
print(df_general['CLUSTER_ID'].value_counts().sort_index())


# ## Analizar resultados de clustering

# Priorice que la dimensión de cada cluster no sea muy extensa o al menos que no supere los 1000 registros.

# In[13]:


print("Resumen de Incidentes por Cluster ---")
cluster_summary = df_general['CLUSTER_ID'].value_counts().sort_index()
print(cluster_summary)

plt.figure(figsize=(14, 7))
sns.countplot(
    data=df_general[df_general['CLUSTER_ID'] != -1], # Excluimos el ruido (-1)
    x='CLUSTER_ID',
)
plt.title('Tamaño de los Hotspots (Clusters)', fontsize=16)
plt.xlabel('ID del Cluster (Zona)', fontsize=12)
plt.ylabel('Cantidad de Incidentes', fontsize=12)
plt.show()


# ## Visualizar clusters

# In[14]:


# Celda 5.6: Mapa de Zonas (Hotspots)
plt.figure(figsize=(12, 12))

# 1. Graficar los clusters (zonas)
sns.scatterplot(
    data=df_general[df_general['CLUSTER_ID'] != -1], # Solo los clusters, no el ruido
    x='LONGITUD',
    y='LATITUD',
    hue='CLUSTER_ID',
    palette='deep',
    legend='full',
    s=20,
    alpha=0.7
)

# 2. Graficar el ruido (puntos grises)
noise_points = df_general[df_general['CLUSTER_ID'] == -1]
plt.scatter(
    noise_points['LONGITUD'],
    noise_points['LATITUD'],
    c='gray',
    label='Ruido (Cluster -1)',
    s=5,
    alpha=0.1
)

plt.title('Mapa de Clusters (Hotspots) de Riesgo General', fontsize=16)
plt.xlabel('Longitud', fontsize=12)
plt.ylabel('Latitud', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# # Construcción del dataset de entrenamiento

# In[15]:


df_model_base = df_general[df_general['CLUSTER_ID'] != -1].copy()

cluster_coords = df_model_base.groupby('CLUSTER_ID')[['LONGITUD', 'LATITUD']].mean().reset_index()

print("\n--- DataFrame base para el modelo ---")
print(df_model_base.head())


# ## Crear plantilla de zona y hora

# In[16]:


zonas_unicas = df_model_base['CLUSTER_ID'].unique()
dias_unicos = range(0, 7) # donde 0=Lunes ... 6=Domingo
horas_unicas = range(0, 24) # del 0 a al 23

df_plantilla = pd.DataFrame(product(zonas_unicas, dias_unicos, horas_unicas), columns=['CLUSTER_ID', 'DIA_SEMANA', 'HORA_DEL_DIA'])

print(f"Total de Zonas: {len(zonas_unicas)}")
print(f"Total de Horas: {len(horas_unicas)}")
print(f"Total de filas en plantilla (Zonas * Horas): {len(df_plantilla)}")
print(df_plantilla.head())


# ## Contar incidentes reales por zona y hora

# In[17]:


df_conteo = df_model_base.groupby(['CLUSTER_ID', 'DIA_SEMANA', 'HORA_DEL_DIA']).size()
df_conteo = df_conteo.reset_index(name='NRO_INCIDENTES')

print("Conteo de incidentes por (Zona, Hora) completado.")
print("\n--- Ejemplo de los conteos ---")
print(df_conteo.head())


# ## Unir plantilla y conteos

# In[18]:


df_entrenamiento = pd.merge(
    df_plantilla,
    df_conteo,
    on=['CLUSTER_ID', 'DIA_SEMANA', 'HORA_DEL_DIA'],
    how='left'
)

df_entrenamiento['NRO_INCIDENTES'] = df_entrenamiento['NRO_INCIDENTES'].fillna(0)

print("Dataset de entrenamiento final (pre-etiquetado) creado.")
print("\n--- Ejemplo del dataset de entrenamiento ---")

print(df_entrenamiento.sample(10, random_state=42))


# # Crear variable nivel de riesgo

# In[19]:


incidentes_positivos = df_entrenamiento[df_entrenamiento['NRO_INCIDENTES'] > 0]['NRO_INCIDENTES']
median_val = incidentes_positivos.median()

print(f"Umbral de corte (Mediana) para Riesgo Medio/Alto: {median_val}")

def asignar_riesgo(n):
    return 'Bajo' if n == 0 else "Alto"

# aplicar la función para crear la columna
df_entrenamiento['NIVEL_RIESGO'] = df_entrenamiento['NRO_INCIDENTES'].apply(asignar_riesgo)

print("Variable objetivo 'NIVEL_RIESGO' creada.")
print("\n--- Ejemplo del dataset con etiquetas ---")
print(df_entrenamiento.sample(10, random_state=1))


# # Visualizacion de desbalance

# Se puede percibir un desbalance en los datos, habiendo mayor cantidad de riesgo "Bajo".

# In[20]:


# Celda 6.6: Gráfico de distribución de la variable objetivo
plt.figure(figsize=(10, 6))
sns.countplot(
    data=df_entrenamiento,
    x='NIVEL_RIESGO',
    order=['Bajo', 'Alto'], # Ordenamos las etiquetas
)
plt.title('Distribución de Clases (Variable Objetivo)', fontsize=16)
plt.xlabel('Nivel de Riesgo', fontsize=12)
plt.ylabel('Cantidad de Registros (Zona-Hora)', fontsize=12)
plt.show()

print("--- Resumen de la Distribución de Clases ---")
print(df_entrenamiento['NIVEL_RIESGO'].value_counts(normalize=True) * 100)


# In[21]:


df_entrenamiento


# # Preparación para el modelo
# 
# Se convertirá las variables categóricas (CLUSTER_ID y HORA_DEL_DIA) en un formato numérico que el modelo pueda usar y separaremos nuestros datos para el entrenamiento.

# In[22]:


feature_cols = ['CLUSTER_ID', 'DIA_SEMANA', 'HORA_DEL_DIA']
target_col = 'NIVEL_RIESGO'

X = df_entrenamiento[feature_cols]
y = df_entrenamiento[target_col]

print(f"Features (X) seleccionadas: {X.columns.tolist()}")
print(f"Target (y) seleccionado: {target_col}")

print("\n--- Ejemplo de X ---")
print(X.head())
print("\n--- Ejemplo de y ---")
print(y.head())


# ## Codificación de features categóricas con One-hot encoding

# In[23]:


encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X)

encoded_feature_names = encoder.get_feature_names_out(feature_cols)

print(f"Datos codificados (One-Hot Encoding) exitosamente.")
print(f"Forma original de X: {X.shape}")
print(f"Nueva forma de X_encoded (features transformadas): {X_encoded.shape}")

print("\n--- Ejemplo de 5 filas de X_encoded ---")
print(X_encoded[:5])
print(f"\nTotal de nuevas features: {len(encoded_feature_names)}")


# ## División de datos para entrenamiento y prueba

# In[24]:


test_size_split = 0.2
random_seed = 42

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=test_size_split,
    random_state=random_seed,
    stratify=y
)

print("Datos divididos en conjuntos de entrenamiento y prueba.")
print(f"Tamaño de X_train (features de entrenamiento): {X_train.shape}")
print(f"Tamaño de y_train (target de entrenamiento): {y_train.shape}")
print(f"Tamaño de X_test (features de prueba): {X_test.shape}")
print(f"Tamaño de y_test (target de prueba): {y_test.shape}")

print("\n--- Verificación de la estratificación (proporciones) ---")
print("Proporción en y (Original):")
print(y.value_counts(normalize=True) * 100)
print("\nProporción en y_train:")
print(y_train.value_counts(normalize=True) * 100)
print("\nProporción en y_test:")
print(y_test.value_counts(normalize=True) * 100)


# # Entrenamiento y evaluación de modelos

# In[25]:


modelos = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    "KNN (k=7)": KNeighborsClassifier(n_neighbors=7),
    "Logistic Regression": LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
    "SVM (Kernel RBF)": SVC(random_state=42, class_weight='balanced'),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

resultados = []
y_pred_gb = None
modelo_gb = None
nombre_gb = "Gradient Boosting"

print(f"--- INICIANDO EVALUACIÓN DE {len(modelos)} MODELOS ---\n")

for nombre, modelo in modelos.items():
    print(f"Entrenando {nombre}...")

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    # como ya se que gradient boosting es el mejor, entonces los resultados de ese algoritmo
    if nombre == nombre_gb:
        y_pred_gb = y_pred
        modelo_gb = modelo

    # metricas
    acc = accuracy_score(y_test, y_pred)
    reporte_dict = classification_report(y_test, y_pred, output_dict=True)
    f1_alto = reporte_dict['Alto']['f1-score']
    recall_alto = reporte_dict['Alto']['recall']

    resultados.append({
        "Modelo": nombre,
        "Accuracy Global": acc,
        "F1-Score (Alto)": f1_alto,
        "Recall (Alto)": recall_alto
    })

if y_pred_gb is not None:
    print(f"\n--- MOSTRANDO MATRIZ DE CONFUSIÓN DE GRADIENT BOOSTING ---")

    fig, ax = plt.subplots(figsize=(7, 6))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred_gb,
        ax=ax,
        cmap='Blues',
        colorbar=False
    )

    f1_gb = [res['F1-Score (Alto)'] for res in resultados if res['Modelo'] == nombre_gb][0]

    ax.set_title(f"{nombre_gb}", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()


print("\n\n--- TABLA COMPARATIVA FINAL ---")
df_resultados = pd.DataFrame(resultados).sort_values(by="F1-Score (Alto)", ascending=False)
print(df_resultados)


# In[ ]:


# obtener modelos
import joblib

joblib.

