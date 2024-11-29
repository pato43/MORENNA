import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from prophet import Prophet
from sklearn.metrics import mean_squared_error

# Configuración inicial
st.set_page_config(page_title="Control de Recursos - PT", layout="wide")

# Título principal del dashboard
st.title("📊 Dashboard de Control de Recursos - Partido del Trabajo (PT)")
st.markdown("""
**Bienvenidos al sistema de control de recursos del PT.**  
Este dashboard permite analizar flujos de costos, identificar anomalías y optimizar la distribución de los recursos con modelos avanzados de Machine Learning.  
- **Objetivo:** Preparar una estructura financiera sólida para la campaña electoral 2027.  
- **Capacidades principales:**
    - Detección de anomalías en gastos.
    - Predicción de flujos de recursos.
    - Clustering y segmentación de gastos.
    - Minería de procesos.
""")

# Función para generar datos simulados
@st.cache_data
def generar_datos():
    np.random.seed(42)
    categorias = ["Salarios", "Propaganda", "Capacitación", "Logística", "Publicidad Digital"]
    meses = np.arange(1, 13)
    data = pd.DataFrame({
        "Categoría": np.random.choice(categorias, 500),
        "Mes": np.random.choice(meses, 500),
        "Gasto ($)": np.abs(np.random.normal(30000, 15000, 500)),
        "Año": np.random.choice([2022, 2023, 2024], 500),
    })
    return data

# Carga de datos
data = generar_datos()

# Barra lateral para filtros
st.sidebar.title("Filtros")
categorias_filtradas = st.sidebar.multiselect("Selecciona categorías", data["Categoría"].unique(), default=data["Categoría"].unique())
anios_filtrados = st.sidebar.multiselect("Selecciona años", data["Año"].unique(), default=data["Año"].unique())

# Filtrar datos según la selección del usuario
data_filtrada = data[(data["Categoría"].isin(categorias_filtradas)) & (data["Año"].isin(anios_filtrados))]

# Verificar si hay datos filtrados
if data_filtrada.empty:
    st.warning("No hay datos disponibles con los filtros seleccionados.")
else:
    # Pestañas principales
    tabs = st.tabs(["📊 Análisis General", "🔎 Detección de Anomalías", "📦 Clustering", "📚 Predicciones Temporales", "🌐 PCA y Segmentación"])

    # --- Pestaña: Análisis General ---
    with tabs[0]:
        st.header("📊 Análisis General")
        col1, col2 = st.columns(2)

        # Gráfico 1: Gasto total por categoría
        gasto_categoria = data_filtrada.groupby("Categoría")["Gasto ($)"].sum().reset_index()
        fig1 = px.bar(gasto_categoria, x="Categoría", y="Gasto ($)", title="Gasto Total por Categoría", text_auto='.2s')
        col1.plotly_chart(fig1, use_container_width=True)

        # Gráfico 2: Gasto mensual promedio
        gasto_mes = data_filtrada.groupby("Mes")["Gasto ($)"].mean().reset_index()
        fig2 = px.line(gasto_mes, x="Mes", y="Gasto ($)", title="Promedio de Gasto Mensual", markers=True)
        col2.plotly_chart(fig2, use_container_width=True)

    # --- Pestaña: Detección de Anomalías ---
    with tabs[1]:
        st.header("🔎 Detección de Anomalías con Isolation Forest")
        st.markdown("""
        Este modelo utiliza técnicas no supervisadas para detectar transacciones inusuales que podrían indicar desviaciones o errores en la gestión.
        """)

        # Modelo de Isolation Forest
        iforest = IsolationForest(contamination=0.05, random_state=42)
        data_filtrada["Anomalía"] = iforest.fit_predict(data_filtrada[["Gasto ($)"]])
        anomalías = data_filtrada[data_filtrada["Anomalía"] == -1]

        if not anomalías.empty:
            fig3 = px.scatter(anomalías, x="Mes", y="Gasto ($)", color="Categoría", title="Anomalías Detectadas")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.success("No se detectaron anomalías en los datos seleccionados.")

    # --- Pestaña: Clustering ---
    with tabs[2]:
        st.header("📦 Clustering de Gastos")
        st.markdown("""
        **K-Means** agrupa los datos en clusters según patrones de gasto. Esto ayuda a entender cómo se distribuyen los recursos.
        """)

        kmeans = KMeans(n_clusters=3, random_state=42)
        data_filtrada["Cluster"] = kmeans.fit_predict(data_filtrada[["Gasto ($)"]])

        fig4 = px.scatter(data_filtrada, x="Mes", y="Gasto ($)", color="Cluster", title="Clustering de Gasto", hover_data=["Categoría"])
        st.plotly_chart(fig4, use_container_width=True)

    # --- Pestaña: Predicciones Temporales ---
    with tabs[3]:
        st.header("📚 Predicciones de Gasto")
        st.markdown("""
        Usando **Prophet**, predecimos los patrones de gasto futuros para una mejor planificación de los recursos.
        """)

        df_prophet = data_filtrada.groupby("Mes")["Gasto ($)"].sum().reset_index()
        df_prophet.columns = ["ds", "y"]

        model = Prophet()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=12, freq="M")
        forecast = model.predict(future)

        fig5 = px.line(forecast, x="ds", y=["yhat", "yhat_lower", "yhat_upper"], title="Predicción de Gasto Mensual")
        st.plotly_chart(fig5, use_container_width=True)

    # --- Pestaña: PCA y Segmentación ---
    with tabs[4]:
        st.header("🌐 Análisis PCA")
        st.markdown("""
        Utilizamos **PCA (Análisis de Componentes Principales)** para reducir la dimensionalidad y facilitar la visualización de los datos.
        """)

        pca = PCA(n_components=2)
        componentes = pca.fit_transform(data_filtrada[["Mes", "Gasto ($)"]])

        fig6 = px.scatter(x=componentes[:, 0], y=componentes[:, 1], color=data_filtrada["Categoría"], title="Reducción de Dimensiones con PCA")
        st.plotly_chart(fig6, use_container_width=True)
