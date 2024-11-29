import streamlit as st
import logging

# Configuración de logging para depuración
logging.basicConfig(
    filename="debug_log.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.debug("Iniciando aplicación Streamlit")

try:
    # Configuración inicial de la aplicación
    st.set_page_config(
        page_title="Dashboard PT",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Título de la aplicación
    st.title("📊 Dashboard de Análisis - Partido del Trabajo (PT)")
    st.sidebar.title("Navegación")
    menu = st.sidebar.radio("Selecciona una opción", ["Inicio", "Datos", "Gráficos"])

    # Página de inicio
    if menu == "Inicio":
        st.subheader("Bienvenido al Dashboard del PT")
        st.write("""
        Este dashboard está diseñado para proporcionar información basada en datos 
        que puede ser utilizada para optimizar decisiones estratégicas del partido.
        """)

    # Página para mostrar datos
    elif menu == "Datos":
        st.subheader("Carga y Visualización de Datos")
        uploaded_file = st.file_uploader("Sube un archivo CSV", type="csv")
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            st.write("Vista previa de los datos cargados:")
            st.write(data.head())
        else:
            st.warning("Por favor, sube un archivo CSV.")

    # Página para gráficos básicos
    elif menu == "Gráficos":
        st.subheader("Gráficos Interactivos")
        example_data = pd.DataFrame({
            "Categoría": ["Salarios", "Propaganda", "Capacitación", "Logística"],
            "Gasto ($)": [20000, 50000, 15000, 25000]
        })
        fig = px.bar(example_data, x="Categoría", y="Gasto ($)", title="Gastos por Categoría")
        st.plotly_chart(fig)

except Exception as e:
    logging.exception("Error en la aplicación Streamlit")
    st.error("Ha ocurrido un error inesperado. Por favor, revisa los logs.")
