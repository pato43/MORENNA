import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from fbprophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Configuración inicial
st.set_page_config(
    page_title="Demo de Dashboard para Optimización de Recursos",
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

# Título y descripción inicial
st.title("🎛️ Demo de Dashboard para Optimización de Recursos")
st.subheader("Automatización y Eficiencia Financiera para Competitividad Electoral 2027")
st.markdown("""
**Objetivo:** Este dashboard permite detectar anomalías, predecir tendencias y optimizar recursos. Diseñado para reducir fugas de dinero, identificar patrones financieros y maximizar el impacto de los recursos en campañas electorales.
""")

# Casos exitosos
st.markdown("""
### 🌍 Casos Exitosos en Otros Países
- **Brasil:** Uso de Machine Learning para monitorear gastos públicos, logrando un ahorro anual de $150 millones de dólares mediante la detección de corrupción en contratos gubernamentales.
- **India:** Implementación de dashboards de gasto público, reduciendo en un 35% los tiempos de procesamiento presupuestario y mejorando la transparencia.
- **Canadá:** Aplicación de herramientas analíticas para predecir desviaciones en proyectos de infraestructura, evitando pérdidas superiores a $50 millones de dólares.
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
    st.header("Opciones de Filtro")
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
        "📦 Clustering de Inventarios (K-Means)", 
        "📚 Predicciones de Gasto (Prophet)", 
        "🌐 PCA para Reducción de Dimensiones", 
        "🌳 Random Forest para Predicción"
    ])

    # --- Pestaña 1: Análisis General ---
    with tabs[0]:
        st.header("📊 Análisis General de Recursos")
        col1, col2 = st.columns(2)

        # Gráfico de gasto por categoría
        fig1 = px.bar(
            data_filtrada.groupby("Categoría")["Gasto ($)"].sum().reset_index(),
            x="Categoría", y="Gasto ($)", color="Categoría",
            title="Gasto Total por Categoría",
            text_auto='.2s'
        )
        fig1.update_traces(textposition='outside', marker=dict(line=dict(color='black', width=1)))
        col1.plotly_chart(fig1, use_container_width=True)

        # Gráfico de gasto mensual
        fig2 = px.line(
            data_filtrada.groupby("Mes")["Gasto ($)"].mean().reset_index(),
            x="Mes", y="Gasto ($)",
            title="Promedio de Gasto Mensual",
            markers=True
        )
        fig2.update_traces(line=dict(width=3), marker=dict(size=10, color="red"))
        col2.plotly_chart(fig2, use_container_width=True)

    # --- Pestaña 2: Transacciones Sospechosas (Isolation Forest) ---
    with tabs[1]:
        st.header("🔎 Transacciones Sospechosas (Isolation Forest)")
        st.markdown("""
        **Objetivo:** Identificar transacciones inusuales que puedan indicar desvíos de recursos o mal manejo financiero.
        """)
        iforest = IsolationForest(contamination=0.05, random_state=42)
        if not data_filtrada.empty:
            data_filtrada["Anomalía"] = iforest.fit_predict(data_filtrada[["Gasto ($)"]])
            anomalías = data_filtrada[data_filtrada["Anomalía"] == -1]
            st.write("Transacciones sospechosas detectadas:", anomalías)
            fig3 = px.scatter(
                anomalías, x="Mes", y="Gasto ($)", color="Categoría",
                title="Transacciones Sospechosas Detectadas"
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("No hay datos suficientes para ejecutar el modelo.")

    # --- Pestaña 3: Clustering de Inventarios ---
    with tabs[2]:
        st.header("📦 Clustering de Inventarios (K-Means)")
        st.markdown("""
        **Objetivo:** Agrupar los gastos en categorías para identificar patrones que puedan indicar fugas de recursos.
        """)
        kmeans = KMeans(n_clusters=3, random_state=42)
        if not data_filtrada.empty:
            data_filtrada["Cluster"] = kmeans.fit_predict(data_filtrada[["Gasto ($)"]])
            fig4 = px.scatter(
                data_filtrada, x="Mes", y="Gasto ($)", color="Cluster",
                title="Clustering de Gasto por Inventarios"
            )
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.warning("No hay datos suficientes para ejecutar el modelo K-Means.")

    # --- Pestaña 4: Predicciones de Gasto con Prophet ---
    with tabs[3]:
        st.header("📚 Predicciones de Gasto (Prophet)")
        st.markdown("""
        **Objetivo:** Utilizar Prophet para predecir tendencias futuras de gasto basándose en datos históricos.
        """)
        df_prophet = data_filtrada[['Mes', 'Gasto ($)']].rename(columns={'Mes': 'ds', 'Gasto ($)': 'y'})
        model = Prophet(yearly_seasonality=True)
        model.fit(df_prophet)
        future = model.make_future_dataframe(df_prophet, periods=12, freq='M')
        forecast = model.predict(future)
        fig5 = px.line(forecast, x='ds', y='yhat', title="Predicción de Gasto con Prophet")
        st.plotly_chart(fig5, use_container_width=True)

    # --- Pestaña 5: PCA ---
    with tabs[4]:
        st.header("🌐 Análisis de Componentes Principales (PCA)")
        st.markdown("""
        **Objetivo:** Reducir la dimensionalidad de los datos para facilitar su interpretación.
        """)
        pca = PCA(n_components=2)
        if not data_filtrada.empty:
            pca_data = pca.fit_transform(data_filtrada[["Mes", "Gasto ($)"]])
            fig6 = px.scatter(
                x=pca_data[:, 0], y=pca_data[:, 1], color=data_filtrada["Categoría"],
                title="Reducción de Dimensiones con PCA"
            )
            st.plotly_chart(fig6, use_container_width=True)
        else:
            st.warning("No hay datos suficientes para ejecutar el PCA.")

    # --- Pestaña 6: Random Forest para Predicción ---
    with tabs[5]:
        st.header("🌳 Random Forest para Predicción")
        st.markdown("""
        **Objetivo:** Utilizar Random Forest para predecir valores de gasto y evaluar su precisión.
        """)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        if not data_filtrada.empty:
            X = data_filtrada[["Mes"]]
            y = data_filtrada["Gasto ($)"]
            rf.fit(X, y)
            y_pred_rf = rf.predict(X)
            mse = mean_squared_error(y, y_pred_rf)
            st.write(f"Error cuadrático medio (MSE): {mse:.2f}")
        else:
            st.warning("No hay datos suficientes para entrenar el modelo Random Forest.")
