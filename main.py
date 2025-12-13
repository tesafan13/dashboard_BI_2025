import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import kagglehub
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import  root_mean_squared_error, r2_score

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
df_2 = pd.read_csv(url)

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
        
        df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
        df.isnull().sum()
        df["children"].isna().sum()
        df.drop(['company'], axis = 1, inplace = True)
        df = df[(df["agent"].notna())  & (df["country"].notna())  & 
        (df["children"].notna()) & (df["adr"] > 0) & (df["adr"] < 5300) 
        & ~((df["children"] > 0) & (df["adults"] == 0)) & 
          ~((df["babies"] > 0) & (df["adults"] == 0)) & 
            ~((df["babies"] > 0) & (df["children"] > 0) & (df["adults"] == 0))]
        df.reset_index(drop=True, inplace=True)
        df.isna().isnull().sum()
        df.isna().sum()
        df_numeric = df.select_dtypes(include='number')
        df_numeric.columns.tolist()
        df_categoric = df.select_dtypes(include='object')
        df_categoric.columns.tolist()
        sns.color_palette("Paired")
        df.drop(columns=['reservation_status', 'reservation_status_date','arrival_date_week_number', 'stays_in_weekend_nights', 'arrival_date_month'], inplace=True)
        df = pd.get_dummies(df, drop_first=True, dtype=int)
        X= df.drop('is_canceled', axis = 1)
        y = df['is_canceled']
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.30, 
        random_state=40)
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=40, stratify=y)
        rf = RandomForestClassifier(n_jobs=-1, random_state=40)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_prob = rf.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, rf_prob)
        auc_rf = roc_auc_score(y_test, rf_prob)
        col1, col2= st.columns(2)
        with col1: 
                plt.figure(figsize=(6,4))
                plt.plot(fpr, tpr, label=f"AUC = {auc_rf:.3f}")
                plt.plot([0,1],[0,1],'--')
                plt.title("ROC Curve - Random Forest")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.legend()
                st.pyplot(plt)
        with col2:
                rf_pred = rf.predict(X_test)
                mat = confusion_matrix(y_test, rf_pred)
                plt.figure(figsize=(6,4))
                sns.heatmap(mat, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["No Cancelado", "Cancelado"],
                yticklabels=["No Cancelado", "Cancelado"])
                plt.xlabel("Predicción")
                plt.ylabel("Real")
                plt.title("Matriz de Confusión - Random Forest")
                st.pyplot(plt)
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
        

        xgb = XGBClassifier(
        n_jobs=-1,
        random_state=40,
        eval_metric='logloss'
                )
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)
        xgb_prob = xgb.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, xgb_prob)
        auc_xgb = roc_auc_score(y_test, xgb_prob)
      
        col1, col2= st.columns(2)
        with col1: 
                plt.figure(figsize=(6,4))
                plt.plot(fpr, tpr, label=f"AUC = {auc_xgb:.3f}")
                plt.plot([0,1],[0,1],'--')
                plt.title("ROC Curve - XGBoost")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.legend()
                st.pyplot(plt)
        with col2:
                cm = confusion_matrix(y_test, xgb_pred)
                plt.figure(figsize=(6,4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
                plt.title("Matriz de Confusión - XGBoost")
                plt.xlabel("Predicción")
                plt.ylabel("Real")
                st.pyplot(plt)
with tab3: 

        df_2.dropna(subset=['children'], inplace=True)
        df_2.dropna(subset=['country'], inplace=True)

        df_2['arrival_date'] = (
                df_2['arrival_date_year'].astype(str) + '-' +
                df_2['arrival_date_month'] + '-' +
                df_2['arrival_date_day_of_month'].astype(str)
                )

        cols = list(df_2.columns)
        cols.insert(3, cols.pop(cols.index('arrival_date')))
        df_2 = df_2.loc[:, cols]

        df_2.drop(
                 ['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month'],
                 axis=1,
                  inplace=True
                )

        df_2['arrival_date'] = pd.to_datetime(df_2['arrival_date'])
        df_2['reservation_status_date'] = pd.to_datetime(df_2['reservation_status_date'])

        df_2['is_canceled'] = df_2['is_canceled'].astype(bool)
        df_2['is_repeated_guest'] = df_2['is_repeated_guest'].astype(bool)
        df_2['children'] = df_2['children'].astype(int)

        df_2[['agent', 'company']].dropna(subset=['agent', 'company'])

        df_2.drop(
                  [
                 "is_canceled", "booking_changes", "reservation_status",
                 "reservation_status_date", "required_car_parking_spaces",
                 "total_of_special_requests", "assigned_room_type",
                 "agent", "company"
                 ],
                 axis=1,
                 inplace=True
                )

        dummy = df_2.drop(columns=['country'])
        dummy = pd.get_dummies(dummy, dtype=int)

        df_2.drop(df_2[df_2['adr'] == 5400].index, inplace=True)
        df_2.drop(df_2[df_2['adr'] < 0].index, inplace=True)

        df_2[df_2['adults'] > 10]
        df_2[df_2['children'] >= 3].sort_values('children', ascending=False)

        df_2.drop(df_2[df_2['children'] == 10].index, inplace=True)
        df_2.drop(df_2[df_2['babies'] >= 3].index, inplace=True)

        df_2[((df_2['adults'] == 0) & (df_2['children'] > 0)) |
         ((df_2['adults'] == 0) & (df_2['babies'] > 0))]

        df_2 = df_2[(df_2['adults'] > 0)]

        df_2[df_2['days_in_waiting_list'] > 0].sort_values(
          'days_in_waiting_list', ascending=False
        )

        df_2[df_2['days_in_waiting_list'] > 0]['days_in_waiting_list'].count()

        months = [
                 'January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December'
                ]

        df_2 = df_2.drop(['arrival_date'], axis=1)

        df_for_model = pd.get_dummies(df_2, drop_first=True, dtype=int)

        X = df_for_model.drop(columns='adr', axis=1)
        y = df_for_model['adr']

        X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
        )

        model_random_forest = RandomForestRegressor()

        model_random_forest.fit(X_train, y_train)

        y_pred = model_random_forest.predict(X_test)
        y_pred_train = model_random_forest.predict(X_train)
        st.header("Modelo RandomForest", divider='green')
        st.subheader("Train", divider = 'gray')
        col1, col2= st.columns(2)
        with col1:
                r2 = 97.4
                st.metric(
                        label="R^2",
                        value=f"{r2}%"
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
                r2 = 84.3
                st.metric(
                        label="R^2",
                        value=f"{r2}%"
                        )
        with col2:
                rmse = 18.99
                st.metric(
                        label="RMSE",
                        value=f"{rmse}"
                        )
        
        plt.figure(figsize=(8,6))
        plt.scatter(y_test, y_pred, alpha=0.3)
        plt.plot(
                 [y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()],
                'r--'
        )

        plt.xlabel("ADR real")
        plt.ylabel("ADR predicho")
        plt.title("Random Forest – Real vs Predicción")
        st.pyplot(plt)

        # Modelo ajustado
        model_random_forest = RandomForestRegressor(
         n_estimators=400,
         max_depth=30,
          min_samples_split=5,
         min_samples_leaf=2,
          max_features=0.7,
          random_state=42,
          n_jobs=-1
        )

        model_random_forest.fit(X_train, y_train)

        y_pred = model_random_forest.predict(X_test)

        y_pred_train = model_random_forest.predict(X_train)
        rmse_train = root_mean_squared_error(y_train, y_pred_train)
        r2_train = r2_score(y_train, y_pred_train)
        st.header("Modelo RandomForest Tuneado", divider='green')
        st.subheader("Train", divider = 'gray')
        col1, col2= st.columns(2)
        with col1:
                r2 = 93.8
                st.metric(
                        label="R^2",
                        value=f"{r2}%"
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
                r2 = 84.5
                st.metric(
                        label="R^2",
                        value=f"{r2}%"
                        )
        with col2:
                rmse = 18.85
                st.metric(
                        label="RMSE",
                        value=f"{rmse}"
                        )
        
        plt.figure(figsize=(8,6))
        plt.scatter(y_test, y_pred, alpha=0.3)
        plt.plot(
                 [y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()],
                 'r--'
                )

        plt.xlabel("ADR real")
        plt.ylabel("ADR predicho")
        plt.title("Random Forest – Real vs Predicción")

        st.pyplot(plt)

