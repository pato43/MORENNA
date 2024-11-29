import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

# Configuración inicial
st.set_page_config(
    page_title="Dashboard de Recursos para Partidos Políticos",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tema de colores
st.markdown("""
<style>
    .css-18e3th9 { background-color: #f9f9f9; } /* Fondo claro */
    .block-container { padding: 1.5rem 2rem; }
    h1, h2, h3 { color: #205375; }
    .stTabs [data-baseweb="tab"] { background-color: #cce7f7; }
</style>
""", unsafe_allow_html=True)

# Título principal
st.title("🎛️ Dashboard de Análisis de Recursos")
st.subheader("Monitoreo, proyección y detección de anomalías en recursos partidistas")
st.markdown("**Nota:** Esta herramienta es una demostración y tiene fines ilustrativos.")

# Carga de datos simulados
@st.cache_data
def load_data():
    np.random.seed(42)
    categories = [
        "Actividades Ordinarias", 
        "Gastos de Proceso Electoral", 
        "Actividades Específicas"
    ]
    data = {
        "Categoría": np.random.choice(categories, 300),
        "Mes": np.random.choice(range(1, 13), 300),
        "Gasto ($)": np.random.randint(10000, 50000, 300),
        "Año": np.random.choice([2022, 2023, 2024], 300),
    }
    return pd.DataFrame(data)

data = load_data()

# Barra lateral
with st.sidebar:
    st.header("Opciones de Filtro")
    filtro_categoria = st.multiselect("Seleccionar Categorías", data["Categoría"].unique(), default=data["Categoría"].unique())
    filtro_año = st.multiselect("Seleccionar Años", data["Año"].unique(), default=data["Año"].unique())

# Filtrar datos
data_filtrada = data[data["Categoría"].isin(filtro_categoria) & data["Año"].isin(filtro_año)]

# Secciones principales
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Análisis General", 
    "🔎 Anomalías", 
    "📈 Proyecciones", 
    "📦 Inventarios"
])

# --- Pestaña 1: Análisis General ---
with tab1:
    st.header("📊 Análisis General de Recursos")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Distribución de Gastos por Categoría")
        fig = px.bar(
            data_filtrada.groupby("Categoría")["Gasto ($)"].sum().reset_index(), 
            x="Categoría", 
            y="Gasto ($)", 
            color="Categoría",
            title="Gastos Totales por Categoría",
            color_discrete_sequence=px.colors.sequential.Blues
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Gasto Promedio por Año")
        fig = px.box(
            data_filtrada, 
            x="Año", 
            y="Gasto ($)", 
            color="Año", 
            title="Distribución de Gastos por Año",
            color_discrete_sequence=px.colors.sequential.Teal
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Pestaña 2: Anomalías ---
with tab2:
    st.header("🔎 Detección de Anomalías en Gastos")
    st.markdown("""
    Este análisis utiliza técnicas de machine learning para identificar gastos inusuales.
    """)
    iforest = IsolationForest(contamination=0.1, random_state=42)
    data_filtrada["Anomalía"] = iforest.fit_predict(data_filtrada[["Gasto ($)"]])
    anomalías = data_filtrada[data_filtrada["Anomalía"] == -1]
    fig = px.scatter(
        data_filtrada, 
        x="Mes", 
        y="Gasto ($)", 
        color="Anomalía", 
        title="Gastos Anómalos Detectados",
        color_discrete_sequence=["#636EFA", "#EF553B"],
        labels={"Anomalía": "Tipo"}
    )
    st.plotly_chart(fig, use_container_width=True)
    if not anomalías.empty:
        st.subheader("Detalles de las Anomalías")
        st.dataframe(anomalías, use_container_width=True)

# --- Pestaña 3: Proyecciones ---
with tab3:
    st.header("📈 Proyecciones Futuras")
    st.markdown("Proyección de gastos basados en tendencias históricas.")
    modelo = LinearRegression()
    proyeccion = data_filtrada.groupby(["Año", "Categoría"])["Gasto ($)"].sum().reset_index()
    for categoria in proyeccion["Categoría"].unique():
        df_cat = proyeccion[proyeccion["Categoría"] == categoria]
        X = df_cat["Año"].values.reshape(-1, 1)
        y = df_cat["Gasto ($)"]
        modelo.fit(X, y)
        proyeccion.loc[proyeccion["Categoría"] == categoria, "Proyección ($)"] = modelo.predict(X)
    fig = px.line(
        proyeccion, 
        x="Año", 
        y="Proyección ($)", 
        color="Categoría", 
        title="Proyecciones de Gasto por Categoría",
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Pestaña 4: Inventarios ---
with tab4:
    st.header("📦 Gestión de Inventarios")
    st.markdown("""
    Monitoreo de inventarios de medicinas, alimentos y gastos operativos.
    """)
    inventarios = {
        "Categoría": ["Medicinas", "Alimentos", "Gastos Operativos"],
        "Disponible": [80, 120, 150],
        "Proyectado (Mes Siguiente)": [60, 100, 140]
    }
    df_inv = pd.DataFrame(inventarios)
    fig = px.bar(
        df_inv, 
        x="Categoría", 
        y=["Disponible", "Proyectado (Mes Siguiente)"], 
        barmode="group", 
        title="Inventarios Actuales y Proyectados",
        color_discrete_sequence=px.colors.sequential.Plasma
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_inv, use_container_width=True)
