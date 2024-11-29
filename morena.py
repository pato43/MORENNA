# Importar librerías necesarias
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import plotly.express as px
from prophet import Prophet

# Configuración inicial de la app
st.set_page_config(page_title="Análisis y Predicciones PT", layout="wide")

# Título de la aplicación
st.title("📊 Análisis Predictivo - Partido del Trabajo (PT)")

# Sidebar para navegación
st.sidebar.title("Opciones de Navegación")
menu = st.sidebar.radio("Selecciona una opción", ("Introducción", "Modelos de ML", "Predicciones Temporales", "Resultados y Visualización"))

# Introducción
if menu == "Introducción":
    st.subheader("Bienvenido")
    st.write("""
    Esta aplicación está diseñada para mostrar cómo las técnicas de análisis de datos y machine learning pueden ser 
    útiles para la toma de decisiones políticas. Exploraremos:
    - Predicción de votantes con regresión logística.
    - Análisis temporal usando Prophet para observar tendencias.
    - Gráficos interactivos para visualizar datos relevantes.
    """)

# Sección de carga de datos
if menu in ["Modelos de ML", "Predicciones Temporales"]:
    st.subheader("Carga de Datos")
    uploaded_file = st.file_uploader("Sube un archivo CSV con los datos", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Vista previa de los datos:")
        st.write(data.head())

# Modelos de Machine Learning
if menu == "Modelos de ML" and uploaded_file is not None:
    st.subheader("Modelo de Regresión Logística")
    
    # Selección de características
    st.write("Selecciona las columnas para el modelo:")
    target = st.selectbox("Columna objetivo", data.columns)
    features = st.multiselect("Características predictoras", [col for col in data.columns if col != target])
    
    if target and features:
        X = data[features]
        y = data[target]
        
        # División de los datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Entrenamiento del modelo
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        # Predicciones
        y_pred = model.predict(X_test)
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Exactitud del modelo: **{accuracy:.2f}**")
        st.text("Reporte de Clasificación")
        st.text(classification_report(y_test, y_pred))
        
        # Matriz de confusión
        st.write("Matriz de Confusión")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                           labels=dict(x="Predicción", y="Real", color="Frecuencia"))
        st.plotly_chart(fig_cm)

# Predicciones Temporales con Prophet
if menu == "Predicciones Temporales" and uploaded_file is not None:
    st.subheader("Análisis de Series Temporales")
    
    # Selección de columnas para Prophet
    st.write("Selecciona las columnas para análisis temporal:")
    date_col = st.selectbox("Columna de Fecha", data.columns)
    value_col = st.selectbox("Columna de Valores", [col for col in data.columns if col != date_col])
    
    if date_col and value_col:
        # Preparar los datos para Prophet
        df_prophet = data[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"})
        
        # Entrenamiento del modelo Prophet
        model = Prophet()
        model.fit(df_prophet)
        
        # Predicciones futuras
        future = model.make_future_dataframe(periods=30)  # Predicción de 30 días adicionales
        forecast = model.predict(future)
        
        # Visualización de resultados
        st.write("Gráfico de Predicciones")
        fig_forecast = px.line(forecast, x="ds", y=["yhat", "yhat_lower", "yhat_upper"],
                               labels={"ds": "Fecha", "value": "Predicción"},
                               title="Predicción de Series Temporales")
        st.plotly_chart(fig_forecast)

# Resultados y Visualización
if menu == "Resultados y Visualización":
    st.subheader("Resumen y Visualización")
    st.write("""
    Este análisis demuestra cómo el machine learning puede ayudar a identificar patrones y predecir tendencias.
    Utiliza herramientas como las regresiones logísticas para clasificar votantes y Prophet para análisis temporal.
    """)
    
    # Gráfico interactivo de muestra
    st.write("Gráfico Interactivo")
    example_data = pd.DataFrame({
        "Categoría": ["A", "B", "C", "D"],
        "Valor": [10, 20, 30, 40]
    })
    fig_example = px.bar(example_data, x="Categoría", y="Valor", title="Ejemplo de Visualización")
    st.plotly_chart(fig_example)
