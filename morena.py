import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

# Configuración inicial
st.set_page_config(
    page_title="Dashboard Partido del Trabajo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tema de colores
st.markdown("""
<style>
    .css-18e3th9 { background-color: #1E1E1E; }
    .block-container { padding: 1.5rem 2rem; }
    h1, h2, h3 { color: #E0E0E0; }
    .stTabs [data-baseweb="tab"] { 
        background-color: #333333;
        color: #E0E0E0;
        border: 1px solid #444444;
        border-radius: 5px;
    }
    .stTabs [data-baseweb="tab"]:hover { 
        background-color: #444444;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { 
        background-color: #205375;
        color: #FFFFFF;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.title("🎛️ Dashboard del Partido del Trabajo")
st.subheader("Monitoreo, proyección y análisis de recursos para el Partido del Trabajo")
st.markdown("""
**Este dashboard integra análisis de gastos, detección de anomalías y proyecciones de presupuesto. Además, ofrecemos los siguientes servicios:**

- **Soporte técnico para comerciantes.**
- **Consultoría en ciencia de datos y minería de procesos** para detectar desvíos de fondos.
- **Educación y capacitación política** enfocada en valores cívicos, derechos humanos y liderazgo femenino.
""")

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
    "💡 Servicios Ofrecidos"
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
    Este análisis utiliza técnicas de machine learning para identificar gastos inusuales. Además, ofrecemos minería de procesos para determinar causas de desvíos de fondos.
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

# --- Pestaña 4: Servicios Ofrecidos ---
with tab4:
    st.header("💡 Servicios Ofrecidos al Partido del Trabajo")
    st.markdown("""
    **Ofrecemos los siguientes servicios especializados para el Partido del Trabajo:**
    - **Soporte técnico para comerciantes:** Soluciones tecnológicas para mejorar la gestión y operación.
    - **Consultoría en ciencia de datos:** Análisis avanzado de datos para optimizar recursos.
    - **Minería de procesos:** Identificación de problemas en el flujo de recursos y detección de desvíos de fondos.
    - **Educación y capacitación política:** Promoción de valores cívicos, derechos humanos y liderazgo político.
    """)
    st.subheader("Gastos en Actividades Ordinarias")
    st.markdown("""
    Incluyen salarios, rentas, gastos de estructura partidista y propaganda institucional, 
    necesarios para el funcionamiento de actividades sectoriales, distritales, municipales, estatales o nacionales.
    """)
    st.subheader("Gastos en Actividades Específicas")
    st.markdown("""
    Enfocados en la educación y capacitación para promover la participación política, 
    valores cívicos y respeto a derechos humanos. También incluye el desarrollo de liderazgo político de las mujeres, 
    asignando al menos el 3% del financiamiento total a este rubro.
    """)
