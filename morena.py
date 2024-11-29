import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configuración inicial
st.set_page_config(
    page_title="Detección y Optimización de Recursos",
    page_icon="📊",
    layout="wide"
)

st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to right, #1e3c72, #2a5298);
        color: white;
    }
    .stButton>button {
        background-color: #f4a261;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar: Configuración de parámetros
st.sidebar.title("Controles Interactivos")
threshold = st.sidebar.slider("Nivel de Anomalía", 0.1, 1.0, 0.5, step=0.1)
clustering_eps = st.sidebar.slider("DBSCAN - Epsilon", 0.1, 2.0, 0.5, step=0.1)
date_range = st.sidebar.date_input("Rango de Fechas", [])
data_choice = st.sidebar.radio("Selecciona el Dataset:", ["Gastos", "Ingresos", "Proyecciones"])

# Carga de datos de ejemplo
st.title("Dashboard de Detección y Optimización de Recursos para MORENA")
st.markdown("**Objetivo:** Identificar y corregir desviaciones financieras, optimizar recursos y fomentar una organización basada en datos.")

@st.cache
def load_data():
    np.random.seed(42)
    data = pd.DataFrame({
        "Fecha": pd.date_range(start="2023-01-01", periods=300),
        "Gasto": np.random.normal(50000, 10000, 300),
        "Ingresos": np.random.normal(60000, 15000, 300),
        "Region": np.random.choice(["Norte", "Sur", "Centro", "Occidente"], 300),
        "Tipo": np.random.choice(["Administrativo", "Operativo", "Campañas"], 300),
    })
    return data

data = load_data()

# Pestañas del dashboard
tab1, tab2, tab3, tab4 = st.tabs(["📈 Análisis General", "🔍 Detección de Anomalías", "📊 Optimización de Gastos", "💡 Proyecciones"])

# Tab 1: Análisis General
with tab1:
    st.header("📈 Análisis General de Recursos")
    st.write("Distribución de gastos e ingresos por región y tipo de recurso.")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(data, values="Gasto", names="Region", title="Distribución de Gastos por Región")
        st.plotly_chart(fig)

    with col2:
        fig = px.bar(data, x="Tipo", y="Ingresos", color="Region", barmode="group", title="Ingresos por Tipo de Recurso")
        st.plotly_chart(fig)

# Tab 2: Detección de Anomalías
with tab2:
    st.header("🔍 Detección de Anomalías en Gastos")
    st.write("Utilizamos Isolation Forest para identificar desviaciones.")

    scaler = StandardScaler()
    iso_forest = IsolationForest(contamination=threshold)
    scaled_data = scaler.fit_transform(data[["Gasto", "Ingresos"]])
    anomalies = iso_forest.fit_predict(scaled_data)
    data["Anomalía"] = np.where(anomalies == -1, "Sí", "No")

    fig = px.scatter(
        data, x="Gasto", y="Ingresos", color="Anomalía",
        title="Detección de Anomalías en Gastos e Ingresos",
        hover_data=["Region", "Tipo"]
    )
    st.plotly_chart(fig)

# Tab 3: Optimización de Gastos
with tab3:
    st.header("📊 Optimización de Gastos")
    st.write("Clusterización de gastos e ingresos para identificar oportunidades de ahorro.")

    dbscan = DBSCAN(eps=clustering_eps, min_samples=5)
    data["Cluster"] = dbscan.fit_predict(scaled_data)

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    cluster_data = pd.DataFrame(pca_data, columns=["PCA1", "PCA2"])
    cluster_data["Cluster"] = data["Cluster"]

    fig = px.scatter(cluster_data, x="PCA1", y="PCA2", color="Cluster", title="Clusterización de Recursos")
    st.plotly_chart(fig)

# Tab 4: Proyecciones
with tab4:
    st.header("💡 Proyecciones de Recursos")
    st.write("Simulación de escenarios futuros en base a datos actuales.")

    if st.button("Generar Proyección"):
        st.write("Proyección generada para los próximos 12 meses:")
        projection = data[["Fecha", "Gasto"]].groupby(pd.Grouper(key="Fecha", freq="M")).sum()
        projection["Proyección"] = projection["Gasto"] * np.random.uniform(0.9, 1.1, len(projection))

        fig = px.line(projection, y=["Gasto", "Proyección"], title="Proyección de Gastos Mensuales")
        st.plotly_chart(fig)
