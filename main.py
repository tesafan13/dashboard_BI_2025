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
        df.dropna(subset=['children'], inplace=True)
        df.dropna(subset=['country'], inplace=True)
        df['arrival_date'] = df['arrival_date_year'].astype(str) + '-' + df['arrival_date_month'] + '-' + df['arrival_date_day_of_month'].astype(str)
        cols = list(df.columns)
        cols.insert(3, cols.pop(cols.index('arrival_date')))
        df = df.loc[:, cols]
        df.drop(['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month'], axis=1, inplace=True)
        df['arrival_date'] = pd.to_datetime(df['arrival_date'])
        df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
        df['is_canceled'] = df['is_canceled'].astype(bool)
        df['is_repeated_guest'] = df['is_repeated_guest'].astype(bool)
        df['children'] = df['children'].astype(int)
        df[['agent', 'company']].dropna(subset=['agent', 'company']).head(20)
        df.drop(["is_canceled",'booking_changes', 'reservation_status', 'reservation_status_date', 'required_car_parking_spaces', 'total_of_special_requests', "assigned_room_type", "agent", "company"], axis=1, inplace=True)
        dummy = df.drop(columns=['country'])
        dummy=pd.get_dummies(dummy, dtype=int)
        df.drop(df[df['adr'] == 5400].index, inplace=True)
        df.drop(df[df['adr'] < 0].index, inplace=True)
        df[df['adults'] > 10]
        df[df['children'] >= 3].sort_values('children', ascending=False)
        df.drop(df[df['children'] == 10].index, inplace=True)
        df.drop(df[df['babies'] >= 3].index, inplace=True)
        df[((df['adults'] == 0) & (df['children'] > 0)) | ((df['adults'] == 0) & (df['babies'] > 0))]
        df = df[(df['adults'] > 0)]
        df[df['days_in_waiting_list'] > 0].sort_values('days_in_waiting_list', ascending=False)
        df[df['days_in_waiting_list'] > 0]['days_in_waiting_list'].count()
        months = ['January', 'February', 'March', 'April', 'May', 'June',
                 'July', 'August', 'September', 'October', 'November', 'December']
        df = df.drop(['arrival_date'], axis=1)
        line_data = df.groupby([
        df['arrival_date'].dt.month_name(),
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

