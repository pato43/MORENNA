# Dashboard para Detección de Procesos y Optimización de Recursos  

Este proyecto es un **dashboard interactivo** desarrollado en **Streamlit**, diseñado para detectar desviaciones en procesos y recursos, así como optimizar gastos mediante el uso de **ciencia de datos** y **machine learning**. El objetivo es implementar soluciones que ayuden al partido político MORENA a identificar anomalías y organizar mejor sus recursos.  

---

## Características Principales  
- **Análisis General**: Visualización de datos por regiones y tipos de recursos usando gráficos dinámicos.  
- **Detección de Anomalías**: Utiliza el algoritmo **Isolation Forest** para identificar desviaciones atípicas en los datos financieros.  
- **Clusterización**: Implementa **DBSCAN** para agrupar datos y analizar patrones de gastos e ingresos.  
- **Proyecciones**: Genera simulaciones y estimaciones para optimizar recursos en el futuro.  
- **Interactividad Avanzada**: Gráficos con zoom, filtros dinámicos y simulaciones basadas en ajustes personalizados.  

---

## Requisitos Previos  
Asegúrate de tener instalado **Python 3.7 o superior** y los siguientes paquetes:  
- `streamlit`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `plotly`
- `scikit-learn`

Puedes instalar las dependencias ejecutando:  
```bash
pip install -r requirements.txt
