from prophet import Prophet
import cmdstanpy

# Configuración del backend
cmdstanpy.install_cmdstan(quiet=True)


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from prophet import Prophet
from sklearn.metrics import mean_squared_error

# Configuración inicial de Streamlit
st.set_page_config(
    page_title="Optimización de Recursos PT",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tema personalizado
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
        background-color: #8B0000;
        color: #FFFFFF;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Título y descripción inicial
st.title("🎛️ Dashboard: Optimización de Recursos para el PT")
st.subheader("Automatización y Eficiencia Financiera en Campañas")
st.markdown("""
Este dashboard está diseñado para ayudar al **Partido del Trabajo (PT)** a gestionar sus recursos de manera más eficiente.  
Se emplean herramientas de **Machine Learning** para detectar anomalías, predecir tendencias de gasto y agrupar categorías clave.  

> **Objetivos Principales:**  
> - Reducir fugas de recursos.  
> - Identificar patrones financieros.  
> - Maximizar el impacto en las campañas electorales.  
""")

# Carga de datos simulados
@st.cache_data
def load_data():
    np.random.seed(42)
    categories = [
        "Salarios", "Administración", "Gastos Médicos", 
        "Limpieza", "Propaganda", "Capacitación"
    ]
    months = np.arange(1, 13)
    fluctuation = np.random.normal(scale=5000, size=500)
    data = {
        "Categoría": np.random.choice(categories, 500),
        "Mes": np.random.choice(months, 500),
        "Gasto ($)": np.abs(np.random.randint(5000, 60000, 500) + fluctuation),
        "Año": np.random.choice([2022, 2023, 2024], 500),
    }
    return pd.DataFrame(data)

data = load_data()

# Barra lateral con filtros
with st.sidebar:
    st.header("📌 Opciones de Filtro")
    filtro_categoria = st.multiselect("Seleccionar Categorías", data["Categoría"].unique(), default=data["Categoría"].unique())
    filtro_año = st.multiselect("Seleccionar Años", data["Año"].unique(), default=data["Año"].unique())

# Filtrar datos
data_filtrada = data.loc[data["Categoría"].isin(filtro_categoria) & data["Año"].isin(filtro_año)].copy()

# Verificar si hay datos filtrados
if data_filtrada.empty:
    st.warning("No hay datos disponibles para los filtros seleccionados.")
else:
    # Pestañas principales
    tabs = st.tabs([
        "📊 Análisis General", 
        "🔎 Transacciones Sospechosas (Isolation Forest)", 
        "📦 Clustering de Gastos (K-Means)", 
        "📚 Predicciones de Gasto (Prophet)", 
        "🌐 Análisis PCA"
    ])

    # --- Pestaña 1: Análisis General ---
    with tabs[0]:
        st.header("📊 Análisis General de Recursos")
        st.markdown("""
        Aquí se presentan gráficos básicos para entender cómo se distribuyen los gastos en diferentes categorías y meses.
        """)

        col1, col2 = st.columns(2)

        # Gráfico 1: Gasto total por categoría
        fig1 = px.bar(
            data_filtrada.groupby("Categoría")["Gasto ($)"].sum().reset_index(),
            x="Categoría", y="Gasto ($)", color="Categoría",
            title="Distribución de Gastos por Categoría",
            text_auto='.2s',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig1.update_traces(textposition='outside', marker=dict(line=dict(color='black', width=1)))
        col1.plotly_chart(fig1, use_container_width=True)

        # Gráfico 2: Promedio de gasto mensual
        fig2 = px.line(
            data_filtrada.groupby("Mes")["Gasto ($)"].mean().reset_index(),
            x="Mes", y="Gasto ($)",
            title="Promedio de Gasto Mensual",
            markers=True,
            line_shape="spline",
            color_discrete_sequence=["#FF4500"]
        )
        fig2.update_traces(line=dict(width=3), marker=dict(size=10))
        col2.plotly_chart(fig2, use_container_width=True)

    # --- Pestaña 2: Transacciones Sospechosas (Isolation Forest) ---
    with tabs[1]:
        st.header("🔎 Transacciones Sospechosas")
        st.markdown("""
        Este modelo usa **Isolation Forest**, una técnica de Machine Learning no supervisado, para identificar gastos inusuales.  
        **Ejemplo:** Una transacción de $60,000 cuando el promedio de la categoría es $10,000 se considera sospechosa.
        """)

        # Aplicar Isolation Forest
        iforest = IsolationForest(contamination=0.05, random_state=42)
        data_filtrada["Anomalía"] = iforest.fit_predict(data_filtrada[["Gasto ($)"]])
        anomalías = data_filtrada[data_filtrada["Anomalía"] == -1]

        if not anomalías.empty:
            fig3 = px.scatter(
                anomalías, x="Mes", y="Gasto ($)", color="Categoría",
                title="Transacciones Sospechosas Detectadas",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.success("No se encontraron anomalías.")

    # --- Pestaña 3: Clustering de Gastos (K-Means) ---
    with tabs[2]:
        st.header("📦 Clustering de Gastos")
        st.markdown("""
        **K-Means** agrupa los gastos en clusters para encontrar patrones ocultos.  
        **Ejemplo:** Gastos de propaganda agrupados según rangos bajos, medios y altos.
        """)

        kmeans = KMeans(n_clusters=3, random_state=42)
        data_filtrada["Cluster"] = kmeans.fit_predict(data_filtrada[["Gasto ($)"]])

        fig4 = px.scatter(
            data_filtrada, x="Mes", y="Gasto ($)", color="Cluster",
            title="Clustering de Gastos por Mes",
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        st.plotly_chart(fig4, use_container_width=True)

    # --- Pestaña 4: Predicciones de Gasto (Prophet) ---
    with tabs[3]:
        st.header("📚 Predicciones de Gasto")
        st.markdown("""
        Usamos **Prophet**, un modelo desarrollado por Facebook, para predecir tendencias de gasto.  
        **Ejemplo:** ¿Cuánto se gastará en "Propaganda" el próximo mes?
        """)

        # Preparar datos para Prophet
        df_prophet = data_filtrada.groupby("Mes")["Gasto ($)"].sum().reset_index()
        df_prophet.columns = ["ds", "y"]

        model = Prophet()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=12, freq='M')
        forecast = model.predict(future)

        fig5 = px.line(
            forecast, x="ds", y="yhat", title="Predicción de Gasto Mensual",
            labels={"ds": "Mes", "yhat": "Gasto ($)"},
            color_discrete_sequence=["#FF4500"]
        )
        st.plotly_chart(fig5, use_container_width=True)

    # --- Pestaña 5: PCA ---
    with tabs[4]:
        st.header("🌐 Análisis PCA")
        st.markdown("""
        **PCA (Análisis de Componentes Principales)** reduce la complejidad de los datos manteniendo patrones clave.  
        **Ejemplo:** Visualizar categorías en un espacio de 2 dimensiones.
        """)

        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(data_filtrada[["Mes", "Gasto ($)"]])

        fig6 = px.scatter(
            x=pca_data[:, 0], y=pca_data[:, 1], color=data_filtrada["Categoría"],
            title="Reducción de Dimensiones con PCA",
            color_discrete_sequence=px.colors.qualitative.D3
        )
        st.plotly_chart(fig6, use_container_width=True)
