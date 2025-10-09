import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
import os
from sklearn.base import BaseEstimator
from sqlalchemy import create_engine, text

# Load model based on target column
def load_model_for_target(target_col):
    model_files = {
        'RB_4800kcal_fob': r"D:\360\Project - 1\Python Codes\lstm_model_RB_4800kcal_fob_best_model.h5",
        'RB_5500kcal_fob': r"D:\360\Project - 1\Python Codes\gru_model_RB_5500kcal_fob_best_model.h5",
        'RB_5700kcal_fob': r"D:\360\Project - 1\Python Codes\linear_regression_model_RB_5700kcal_fob_best_model.pkl",
        'India_5500kcal_cfr': r"D:\360\Project - 1\Python Codes\gru_model_India_5500kcal_cfr_best_model.h5",
        'RB_6000kcal_avg': r"D:\360\Project - 1\Python Codes\lstm_model_RB_6000kcal_avg_best_model.h5"
    }

    model_file = model_files.get(target_col)

    if not model_file:
        st.error("No model file configured for the selected target column.")
        return None

    if not os.path.exists(model_file):
        st.error(f"Model file not found at path: {model_file}")
        return None

    try:
        if model_file.endswith('.pkl'):
            model = joblib.load(model_file)
        elif model_file.endswith('.h5'):
            model = load_model(model_file, compile=False)
        else:
            st.error("Unsupported model file format.")
            return None
    except Exception as e:
        st.error(f"Error loading model for {target_col}: {e}")
        return None

    return model

# Forecasting
def make_forecast(model, data, n_days=15):
    numeric_data = data.select_dtypes(include=[np.number])

    if isinstance(model, BaseEstimator):  # sklearn model (e.g., linear regression)
        try:
            if numeric_data.shape[1] != model.n_features_in_:
                st.error(f"Feature mismatch! Model expects {model.n_features_in_} features, got {numeric_data.shape[1]}.")
                return None
            forecast_data = numeric_data.tail(n_days)
            forecast = model.predict(forecast_data)
        except Exception as e:
            st.error(f"Error during sklearn forecast: {e}")
            return None

    elif hasattr(model, 'predict'):  # LSTM/GRU
        try:
            if numeric_data.shape[0] >= 7:
                data_for_prediction = numeric_data.iloc[-7:].values

                if data_for_prediction.shape[1] == 16:
                    data_for_prediction = np.reshape(data_for_prediction, (1, 7, 16))
                elif data_for_prediction.shape[1] < 16:
                    missing = 16 - data_for_prediction.shape[1]
                    padded = np.pad(data_for_prediction, ((0, 0), (0, missing)), mode='constant')
                    data_for_prediction = np.reshape(padded, (1, 7, 16))
                else:
                    st.error("Too many features (expected max 16).")
                    return None

                forecast = []
                for _ in range(n_days):
                    prediction = model.predict(data_for_prediction)
                    forecast.append(prediction[0][0])
                    data_for_prediction = np.roll(data_for_prediction, shift=-1, axis=1)
                    data_for_prediction[0, -1, -1] = prediction
            else:
                st.error("Need at least 7 rows for LSTM/GRU.")
                return None
        except Exception as e:
            st.error(f"Error during deep learning forecast: {e}")
            return None
    else:
        st.error("Unsupported model type.")
        return None

    return forecast

# Plot forecast line chart
def plot_forecast(forecast, target_col):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(forecast)+1), forecast, marker='o', linestyle='--', color='teal')
    ax.set_title(f'Forecast for {target_col}')
    ax.set_xlabel('Day')
    ax.set_ylabel('Forecasted Value')
    ax.grid(True)
    st.pyplot(fig)

# Forecast Dashboard Charts
def display_dashboard_charts(forecast, target_col):
    forecast_series = pd.Series(forecast, name='Forecast')
    forecast_df = pd.DataFrame({
        'Day': list(range(1, len(forecast)+1)),
        'Forecast': forecast_series
    })

    st.subheader("📊 Forecast Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("📈 Mean", f"{forecast_series.mean():.2f}")
    col2.metric("🔽 Min", f"{forecast_series.min():.2f}")
    col3.metric("🔼 Max", f"{forecast_series.max():.2f}")

    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.markdown("**📊 Bar Chart**")
        st.bar_chart(forecast_series)

    with row1_col2:
        st.markdown("**📈 Area Chart**")
        st.area_chart(forecast_series)

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.markdown("**📉 Histogram**")
        fig1, ax1 = plt.subplots()
        ax1.hist(forecast, bins=10, color='skyblue', edgecolor='black')
        ax1.set_title('Histogram of Forecasted Values')
        st.pyplot(fig1)

    with row2_col2:
        st.markdown("**🥧 Pie Chart**")
        bins = pd.cut(forecast_series, bins=5)
        pie_data = bins.value_counts().sort_index()
        fig2, ax2 = plt.subplots()
        ax2.pie(pie_data, labels=[str(c) for c in pie_data.index], autopct='%1.1f%%')
        ax2.set_title('Forecast Distribution')
        st.pyplot(fig2)

    st.markdown("**🌡️ Heatmap (Correlation with Day)**")
    corr_matrix = forecast_df.corr()
    fig3, ax3 = plt.subplots()
    cax = ax3.matshow(corr_matrix, cmap='coolwarm')
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    fig3.colorbar(cax)
    st.pyplot(fig3)

# SQL Connection Setup
def get_sql_connection(user, pw, db):
    try:
        engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
        return engine
    except Exception as e:
        st.error(f"Error connecting to SQL database: {e}")
        return None

# --- Streamlit UI ---
st.set_page_config(page_title="Coal Price Forecast", layout="wide")
st.title('📊 Coal Price Forecasting Dashboard')

# Sidebar Inputs
target_col = st.sidebar.selectbox('Select Target Column', [
    'RB_4800kcal_fob',
    'RB_5500kcal_fob',
    'RB_5700kcal_fob',
    'India_5500kcal_cfr',
    'RB_6000kcal_avg'
])

uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV/XLSX)", type=["csv", "xlsx"])

forecast_days = st.sidebar.slider("Select Forecast Days", min_value=1, max_value=30, value=15)

# SQL Connection Inputs
st.sidebar.subheader("SQL Connection")
sql_user = st.sidebar.text_input("Username", value="root")
sql_pw = st.sidebar.text_input("Password", type="password")
sql_db = st.sidebar.text_input("Database Name", value="coal")
sql_table = st.sidebar.text_input("Forecast Table Name", value="forecast_results")

tab1, tab2 = st.tabs(["📊 Data Preview", "📈 Forecast"])

with tab1:
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                data = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format.")
                data = None

            if data is not None:
                st.subheader("📄 Uploaded Dataset")
                st.dataframe(data.head())
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("Please upload a dataset to begin.")

with tab2:
    if uploaded_file is not None:
        model = load_model_for_target(target_col)

        if model:
            st.success(f"✅ Model loaded for: {target_col}")
            forecast = make_forecast(model, data, n_days=forecast_days)

            if forecast is not None:
                st.subheader(f"📈 Forecast Result (Next {forecast_days} Days)")
                forecast_df = pd.DataFrame({
                    'Day': list(range(1, len(forecast)+1)),
                    f'{target_col}_Forecast': forecast
                })
                forecast_df['Date'] = pd.to_datetime('today').strftime('%Y-%m-%d')
                st.dataframe(forecast_df)

                plot_forecast(forecast, target_col)
                display_dashboard_charts(forecast, target_col)

                if st.button('Save Forecast to SQL'):
                    engine = get_sql_connection(sql_user, sql_pw, sql_db)
                    if engine:
                        try:
                            with engine.connect() as conn:
                                forecast_col = f"{target_col}_Forecast"
                                create_stmt = f"""
                                CREATE TABLE IF NOT EXISTS {sql_table} (
                                    Day INT,
                                    Date DATE,
                                    {forecast_col} FLOAT
                                )
                                """
                                conn.execute(text(create_stmt))
                                forecast_df.to_sql(sql_table, con=engine, if_exists='append', index=False)
                                st.success("✅ Forecast saved to SQL successfully!")
                        except Exception as e:
                            st.error(f"Error saving forecast to SQL: {e}")
