import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

# Configuración inicial
st.set_page_config(page_title="Análisis de Recursos para Partidos Políticos", layout="wide")

# Título principal
st.title("Análisis Integral de Recursos para Partidos Políticos")
st.markdown("""
**Esta es una herramienta de demostración** para el análisis, monitoreo y gestión de recursos económicos, operativos, inventarios y categorías de gastos. 
El sistema permite evaluar patrones de uso, detectar anomalías y realizar proyecciones para una mejor toma de decisiones.
""")

# Carga de datos simulados
@st.cache
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

# Pestañas principales
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔎 Análisis General", 
    "📊 Anomalías", 
    "📈 Proyecciones", 
    "📦 Inventarios", 
    "📜 Categorías de Gastos"
])

# --- Pestaña 1: Análisis General ---
with tab1:
    st.header("Análisis General de Recursos")
    st.markdown("Visualización de gastos por áreas e instancias.")
    
    col1, col2 = st.columns(2)
    with col1:
        category_gastos = data.groupby("Categoría")["Gasto ($)"].sum().reset_index()
        fig = px.bar(category_gastos, x="Categoría", y="Gasto ($)", title="Gasto Total por Categoría")
        st.plotly_chart(fig)
    
    with col2:
        gasto_mes = data.groupby("Mes")["Gasto ($)"].sum().reset_index()
        fig = px.line(gasto_mes, x="Mes", y="Gasto ($)", title="Tendencia Mensual de Gastos")
        st.plotly_chart(fig)

# --- Pestaña 2: Anomalías ---
with tab2:
    st.header("Detección de Anomalías en los Gastos")
    st.markdown("Identificación de gastos atípicos utilizando machine learning.")
    
    iforest = IsolationForest(contamination=0.1, random_state=42)
    data["Anomalía"] = iforest.fit_predict(data[["Gasto ($)"]])
    
    anomalías = data[data["Anomalía"] == -1]
    fig = px.scatter(data, x="Mes", y="Gasto ($)", color="Anomalía", title="Anomalías en los Gastos")
    st.plotly_chart(fig)
    st.dataframe(anomalías, use_container_width=True)

# --- Pestaña 3: Proyecciones ---
with tab3:
    st.header("Proyecciones de Gastos Futuros")
    st.markdown("Proyección del uso de recursos en categorías clave.")
    
    proyeccion = data.groupby(["Año", "Categoría"])["Gasto ($)"].sum().reset_index()
    modelo = LinearRegression()
    for categoria in proyeccion["Categoría"].unique():
        df_cat = proyeccion[proyeccion["Categoría"] == categoria]
        X = df_cat["Año"].values.reshape(-1, 1)
        y = df_cat["Gasto ($)"]
        modelo.fit(X, y)
        proyeccion.loc[proyeccion["Categoría"] == categoria, "Proyección ($)"] = modelo.predict(X)
    
    fig = px.line(proyeccion, x="Año", y="Proyección ($)", color="Categoría", title="Proyección de Gastos por Categoría")
    st.plotly_chart(fig)

# --- Pestaña 4: Inventarios ---
with tab4:
    st.header("Gestión Eficiente de Inventarios")
    st.markdown("Visualización de inventarios de medicinas, alimentos y gastos operativos.")
    
    inventarios = {
        "Categoría": ["Medicinas", "Alimentos", "Gastos Operativos"],
        "Disponible": [80, 120, 150],
        "Proyectado (Mes Siguiente)": [60, 100, 140]
    }
    df_inv = pd.DataFrame(inventarios)
    fig = px.bar(df_inv, x="Categoría", y=["Disponible", "Proyectado (Mes Siguiente)"], barmode="group", title="Inventarios Actuales y Proyectados")
    st.plotly_chart(fig)
    st.dataframe(df_inv, use_container_width=True)

# --- Pestaña 5: Categorías de Gastos ---
with tab5:
    st.header("Análisis Detallado por Categorías de Gastos")
    st.markdown("""
    - **Actividades Ordinarias:** Incluyen salarios, rentas, gastos de estructura partidista y propaganda institucional.
    - **Gastos de Proceso Electoral:** Propaganda electoral, publicidad, eventos públicos, y producción de mensajes.
    - **Actividades Específicas:** Promoción de participación política, valores cívicos y derechos humanos, con énfasis en liderazgo femenino.
    """)
    
    fig = px.sunburst(
        data, 
        path=["Categoría", "Mes"], 
        values="Gasto ($)", 
        title="Distribución de Gastos por Categoría y Mes"
    )
    st.plotly_chart(fig)
