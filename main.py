import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title = "Dashboard Ventas Tienda Tech",
    page_icon = "✨",
    layout="wide",
    initial_sidebar_state="expanded"

)
url = 'https://raw.githubusercontent.com/JohnChirinos/BI/refs/heads/main/hotel_bookings.csv'
@st.cache_data
def load_data(url):
        df = pd.read_csv(url)
        return df


st.title("Optimización de Ingresos en    la Industria Hotelera")
tab1, tab2, tab3 = st.tabs(["👋 Introducción", "❌ Cancelaciones", "💵 Optimización de Ingresos"])
with tab1:
        st.subheader("Introducción")
        st.write("""En este proyecto buscamos desarrollar dos modelos predictivos que permita mejorar las estrategias de gestión de ingresos tanto en hoteles urbanos como en hoteles tipo resort. Se busca identificar oportunidades concretas para incrementar ingresos, reducir cancelaciones y optimizar la ocupación.
        """)
        st.markdown("""Utilizamos el siguiente dataset: [Hotel Booking Demand Dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
        """)
        df = load_data(url)
        st.dataframe(df.head(30))
        st.caption("Nota: Primeros 30 valores del archivo CSV")
        st.write("""En las columnas se pueden observar datos como el tipo de hotel, reservaciones, cancelaciones, fechas y tiempos de llegada, permisos, el páis de origen, tipo de cuarto, forma de pago, y otros datos. 
        """)
with tab2:
        st.header("❌ Cancelaciones", divider = 'red')
        st.header("Modelo Random Forest", divider='green')
        st.subheader("Predice que no cancelan", divider = 'gray')
        col1, col2, col3= st.columns(3)
        with col1:
                precision_valor = 84
                st.metric(
                        label="Precisión",
                        value=f"{precision_valor}%"
                        )
        with col2:
                recall_v = 88
                st.metric(
                        label="Recall",
                        value=f"{recall_v}%"
                        )
        with col3:
                score_v = 86
                st.metric(
                        label="f1_score",
                        value=f"{score_v}%"
                        )
        st.subheader("Predice que cancelan", divider = 'gray')
        col1, col2, col3= st.columns(3)
        with col1:
                precision = 81
                st.metric(
                        label="Precisión",
                        value=f"{precision}%"
                        )
                
        with col2:
                recall = 79
                st.metric(
                        label="Recall",
                        value=f"{recall}%"
                        )
        with col3:
                score = 77
                st.metric(
                        label="f1_score",
                        value=f"{score}%"
                        )
        col1, col2= st.columns(2)
        with col1: 
                st.image("roc_random.png")
        with col2:
                st.image("mat_conf_randomf.png")
        st.header("Modelo XGBoost", divider='green')
        st.subheader("Predice que no cancelan", divider = 'gray')
        col1, col2, col3= st.columns(3)
        with col1:
                precision_valor = 89
                st.metric(
                        label="Precisión",
                        value=f"{precision_valor}%"
                        )
        with col2:
                recall_v = 90
                st.metric(
                        label="Recall",
                        value=f"{recall_v}%"
                        )
        with col3:
                score_v = 89
                st.metric(
                        label="f1_score",
                        value=f"{score_v}%"
                        )
        st.subheader("Predice que cancelan", divider = 'gray')
        col1, col2, col3= st.columns(3)
        with col1:
                precision = 85
                st.metric(
                        label="Precisión",
                        value=f"{precision}%"
                        )
                
        with col2:
                recall = 82
                st.metric(
                        label="Recall",
                        value=f"{recall}%"
                        )
        with col3:
                score = 83
                st.metric(
                        label="f1_score",
                        value=f"{score}%"
                        )      
        col1, col2= st.columns(2)
        with col1: 
                st.image("roc_xg.png")
        with col2:
                st.image("mat_conf_xg.png")
with tab3: 
        st.header("Modelo RandomForest", divider='green')
        st.subheader("Train", divider = 'gray')
        col1, col2= st.columns(2)
        with col1:
                r2 = .974
                st.metric(
                        label="R^2",
                        value=f"{r2}"
                        )
        with col2:
                rmse = 7.81
                st.metric(
                        label="RMSE",
                        value=f"{rmse}"
                        )
        st.subheader("Test", divider = 'gray')
        col1, col2= st.columns(2)
        with col1:
                r2 = .843
                st.metric(
                        label="R^2",
                        value=f"{r2}"
                        )
        with col2:
                rmse = 18.99
                st.metric(
                        label="RMSE",
                        value=f"{rmse}"
                        )
        
        st.image("real_pred.png")
        st.header("Modelo RandomForest Tuneado", divider='green')
        st.subheader("Train", divider = 'gray')
        col1, col2= st.columns(2)
        with col1:
                r2 = .938
                st.metric(
                        label="R^2",
                        value=f"{r2}"
                        )
        with col2:
                rmse = 11.91
                st.metric(
                        label="RMSE",
                        value=f"{rmse}"
                        )
        st.subheader("Test", divider = 'gray')
        col1, col2= st.columns(2)
        with col1:
                r2 = .845
                st.metric(
                        label="R^2",
                        value=f"{r2}"
                        )
        with col2:
                rmse = 18.85
                st.metric(
                        label="RMSE",
                        value=f"{rmse}"
                        )
        
        st.image("real_pred_tune.png")

