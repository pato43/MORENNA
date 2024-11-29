import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Configuración inicial
st.set_page_config(
    page_title="Demo de Algoritmos de ML - Partido del Trabajo",
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
st.title("🎛️ Demo de Algoritmos de Machine Learning - Partido del Trabajo")
st.subheader("Optimización de Recursos y Competitividad Electoral 2027")
st.markdown("""
**Objetivo:** Presentar cómo los algoritmos avanzados de Machine Learning pueden identificar problemas financieros, mejorar la gestión de recursos y detectar anomalías para destinar el dinero recuperado a la competitividad electoral.
""")

# Carga de datos simulados
@st.cache_data
def load_data():
    np.random.seed(42)
    categories = [
        "Salarios", "Administración", "Gastos Médicos", 
        "Limpieza", "Propaganda", "Capacitación"
    ]
    data = {
        "Categoría": np.random.choice(categories, 500),
        "Mes": np.random.choice(range(1, 13), 500),
        "Gasto ($)": np.random.randint(5000, 60000, 500),
        "Año": np.random.choice([2022, 2023, 2024], 500),
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

# Pestañas principales
tabs = st.tabs([
    "📊 Análisis General", 
    "🔎 Transacciones Sospechosas (Isolation Forest)", 
    "📦 Clustering de Inventarios (K-Means)", 
    "📚 Predicciones de Gasto (Regresión Lineal)", 
    "🌟 XGBoost para Clasificación", 
    "🌐 PCA para Reducción de Dimensiones", 
    "🌳 Random Forest para Predicción",
    "📘 Teoría de Algoritmos"
])

# --- Pestaña 1: Análisis General ---
with tabs[0]:
    st.header("📊 Análisis General de Recursos")
    col1, col2 = st.columns(2)
    
    # Gráfico de gasto por categoría
    fig1 = px.bar(
        data_filtrada.groupby("Categoría")["Gasto ($)"].sum().reset_index(),
        x="Categoría", y="Gasto ($)", color="Categoría",
        title="Gasto Total por Categoría"
    )
    col1.plotly_chart(fig1, use_container_width=True)
    
    # Gráfico de gasto mensual
    fig2 = px.line(
        data_filtrada.groupby("Mes")["Gasto ($)"].sum().reset_index(),
        x="Mes", y="Gasto ($)", title="Gasto Mensual"
    )
    col2.plotly_chart(fig2, use_container_width=True)

# --- Pestaña 2: Transacciones Sospechosas ---
with tabs[1]:
    st.header("🔎 Transacciones Sospechosas (Isolation Forest)")
    st.markdown("""
    **Objetivo:** Identificar transacciones inusuales que puedan indicar desvíos de recursos o mal manejo financiero.
    """)
    # Aplicar Isolation Forest para detectar anomalías
    iso_model = IsolationForest(contamination=0.05, random_state=42)
    data_filtrada["Anomalía"] = iso_model.fit_predict(data_filtrada[["Gasto ($)"]])
    anomalías = data_filtrada[data_filtrada["Anomalía"] == -1]
    
    st.write("Transacciones sospechosas detectadas:", anomalías)
    fig3 = px.scatter(
        anomalías, x="Mes", y="Gasto ($)", color="Categoría",
        title="Transacciones Sospechosas Detectadas"
    )
    st.plotly_chart(fig3, use_container_width=True)

# --- Pestaña 3: Clustering de Inventarios ---
with tabs[2]:
    st.header("📦 Clustering de Inventarios (K-Means)")
    st.markdown("""
    **Objetivo:** Agrupar los gastos en categorías para identificar patrones que puedan indicar fugas de recursos.
    """)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    data_filtrada["Cluster"] = kmeans.fit_predict(data_filtrada[["Gasto ($)"]])
    
    fig4 = px.scatter(
        data_filtrada, x="Mes", y="Gasto ($)", color="Cluster",
        title="Clustering de Gasto por Inventarios"
    )
    st.plotly_chart(fig4, use_container_width=True)

# --- Pestaña 4: Predicciones de Gasto ---
with tabs[3]:
    st.header("📚 Predicciones de Gasto (Regresión Lineal)")
    st.markdown("""
    **Objetivo:** Predecir tendencias futuras de gasto basándose en datos históricos.
    """)
    lr = LinearRegression()
    X = data_filtrada[["Mes"]]
    y = data_filtrada["Gasto ($)"]
    lr.fit(X, y)
    predicciones = lr.predict(X)
    data_filtrada["Predicción ($)"] = predicciones
    
    fig5 = px.line(
        data_filtrada, x="Mes", y="Predicción ($)", color="Categoría",
        title="Predicciones de Gasto con Regresión Lineal"
    )
    st.plotly_chart(fig5, use_container_width=True)

# --- Pestaña 5: XGBoost para Clasificación ---
with tabs[4]:
    st.header("🌟 XGBoost para Clasificación")
    st.markdown("""
    **Objetivo:** Utilizar XGBoost para clasificar gastos sospechosos según su categoría.
    """)
    X_train, X_test, y_train, y_test = train_test_split(data_filtrada[["Mes", "Gasto ($)"]], data_filtrada["Categoría"], test_size=0.3, random_state=42)
    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Precisión del modelo XGBoost: {accuracy:.2f}")

# --- Pestaña 6: PCA ---
with tabs[5]:
    st.header("🌐 Análisis de Componentes Principales (PCA)")
    st.markdown("""
    **Objetivo:** Reducir la dimensionalidad de los datos para facilitar su interpretación.
    """)
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data_filtrada[["Mes", "Gasto ($)"]])
    fig6 = px.scatter(
        x=pca_data[:, 0], y=pca_data[:, 1], color=data_filtrada["Categoría"],
        title="Reducción de Dimensiones con PCA"
    )
    st.plotly_chart(fig6, use_container_width=True)

# --- Pestaña 7: Random Forest ---
with tabs[6]:
    st.header("🌳 Random Forest para Predicción")
    st.markdown("""
    **Objetivo:** Utilizar Random Forest para predecir valores de gasto y evaluar su precisión.
    """)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    y_pred_rf = rf.predict(X)
    mse = mean_squared_error(y, y_pred_rf)
    st.write(f"Error cuadrático medio (MSE): {mse:.2f}")
