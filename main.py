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


st.title("Optimización de Ingresos en la Industria Hotelera y Cancelaciones")
tab1, tab2, tab3 = st.tabs(["👋 Introducción", "❌ Cancelaciones", "💵 Optimización de Ingresos"])
with tab1:
        st.subheader("Introducción")
        st.write("""En este proyecto buscamos desarrollar dos modelos predictivos que permita mejorar las estrategias de gestión de ingresos tanto en hoteles urbanos como en hoteles tipo resort. Se busca identificar oportunidades concretas para incrementar ingresos, reducir cancelaciones y optimizar la ocupación.
        """)
        st.markdown("""Utilizamos el siguiente dataset: [Hotel Booking Demand Dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
        """)
        df = load_data(url)
        st.dataframe(df)
        st.caption("Nota: Primeros 30 valores del archivo CSV")
        st.write("""En las columnas se pueden observar datos como el tipo de hotel, reservaciones, cancelaciones, fechas y tiempos de llegada, permisos, el páis de origen, tipo de cuarto, forma de pago, y otros datos. 
        """)
        df_chart = df.copy()
        df_chart.dropna(subset=['children'], inplace=True)
        df_chart.dropna(subset=['country'], inplace=True)
    
        # Crear la columna arrival_date
        df_chart['arrival_date'] = df_chart['arrival_date_year'].astype(str) + '-' + df_chart['arrival_date_month'] + '-' + df_chart['arrival_date_day_of_month'].astype(str)
        # ... (El resto de la lógica de transformación y limpieza que NO AFECTE A df original) ...
        df_chart['arrival_date'] = pd.to_datetime(df_chart['arrival_date'])
        months = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']
        # El resto del código de la gráfica usa df_chart:
        line_data = df_chart.groupby([
            df_chart['arrival_date'].dt.month_name(),
            'hotel'
        ])['adr'].mean().reset_index()
    
        line_data['month'] = pd.Categorical(
                line_data['arrival_date'],
                categories=months,
                ordered=True
            )

        line_data = line_data.sort_values('month')

        fig, ax1 = plt.subplots(figsize=(12, 8))

        sns.lineplot(
            line_data,
            x='month',
            y='adr',
            hue='hotel',
            marker='o',
            ax=ax1
        )

        ax1.set_title("Tarifa diaria promedio por mes y tipo de hotel", fontsize=20, pad=25)
        ax1.set_xlabel("Mes", fontsize=20, labelpad=18)
        ax1.set_ylabel("Tarifa diaria promedio (ADR)", fontsize=16, labelpad=15)
        ax1.tick_params(axis='x', rotation=18, labelsize=12)
        ax1.tick_params(axis='y', labelsize=14)
        ax1.legend(title="Tipo de hotel")

        plt.tight_layout()
        st.pyplot(fig)
        
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

