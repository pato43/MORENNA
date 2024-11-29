import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Configuración de la página
st.set_page_config(
    page_title="Demo de Análisis de Recursos - Partido Politico",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Título y descripción
st.title("Demo de Análisis de Recursos y Optimización - Partido Politico 💼")
st.markdown("""
Este dashboard es una **demo** interactiva para visualizar y analizar el gasto de diferentes áreas del partido. 
Incluye análisis de detección de anomalías y sistemas de optimización basados en ciencia de datos y machine learning.
""")

# Cargar datos (simulados)
np.random.seed(42)
data = pd.DataFrame({
    "Área": np.random.choice(["Administración", "Campañas", "Logística", "Publicidad", "Operaciones", "Consultorías"], 100),
    "Monto (millones)": np.random.uniform(1, 100, 100),
    "Mes": np.random.choice(["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio"], 100),
    "Año": np.random.choice([2023, 2024], 100),
    "Instancia": np.random.choice(["Nacional", "Regional", "Estatal"], 100)
})

# Sidebar para filtros
st.sidebar.header("Filtros")
selected_area = st.sidebar.multiselect("Selecciona Área(s):", options=data["Área"].unique(), default=data["Área"].unique())
selected_year = st.sidebar.selectbox("Selecciona Año:", options=data["Año"].unique())
selected_instance = st.sidebar.multiselect("Selecciona Instancia(s):", options=data["Instancia"].unique(), default=data["Instancia"].unique())

filtered_data = data[
    (data["Área"].isin(selected_area)) &
    (data["Año"] == selected_year) &
    (data["Instancia"].isin(selected_instance))
]

# Gráficos principales
st.markdown("### Análisis General de Gasto por Área")
fig_area = px.bar(filtered_data, x="Área", y="Monto (millones)", color="Área", title="Gasto por Área", barmode="group")
st.plotly_chart(fig_area, use_container_width=True)

st.markdown("### Gasto Total por Mes")
fig_month = px.line(filtered_data, x="Mes", y="Monto (millones)", color="Instancia", title="Gasto por Mes e Instancia", markers=True)
st.plotly_chart(fig_month, use_container_width=True)

# Detección de Anomalías
st.markdown("### Detección de Anomalías en los Gastos")

# Preparar datos para el modelo
scaler = StandardScaler()
scaled_data = scaler.fit_transform(filtered_data[["Monto (millones)"]])

# Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
filtered_data["Anomalía"] = iso_forest.fit_predict(scaled_data)
anomalies = filtered_data[filtered_data["Anomalía"] == -1]

# Mostrar anomalías en gráfico
fig_anomalies = px.scatter(filtered_data, x="Monto (millones)", y="Área", color="Anomalía",
                           title="Detección de Anomalías en Gastos",
                           color_discrete_map={-1: "red", 1: "blue"})
st.plotly_chart(fig_anomalies, use_container_width=True)

# Clusterización (DBSCAN)
st.markdown("### Agrupación de Gastos por DBSCAN")
dbscan = DBSCAN(eps=1, min_samples=5)
filtered_data["Cluster"] = dbscan.fit_predict(scaled_data)

fig_clusters = px.scatter(filtered_data, x="Monto (millones)", y="Área", color="Cluster",
                           title="Clusterización de Gastos",
                           color_continuous_scale="Viridis")
st.plotly_chart(fig_clusters, use_container_width=True)

# Dashboard de simulación
st.markdown("### Simulación y Proyecciones de Gasto")
projection_multiplier = st.slider("Ajuste de Incremento de Gasto (%)", min_value=0, max_value=100, value=10)
filtered_data["Proyección"] = filtered_data["Monto (millones)"] * (1 + projection_multiplier / 100)

fig_projection = px.area(filtered_data, x="Mes", y="Proyección", color="Área",
                         title=f"Proyección de Gasto con Incremento del {projection_multiplier}%")
st.plotly_chart(fig_projection, use_container_width=True)

# Descripciones adicionales
st.markdown("""
**Nota:** Esta demo utiliza datos simulados. En un entorno real, las proyecciones y modelos se ajustan con datos reales y criterios definidos 
por expertos en gestión financiera y analistas de datos.
""")
