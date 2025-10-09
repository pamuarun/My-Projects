# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 21:46:54 2025

@author: Arun Teja
"""

#Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import lag_plot
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, ElasticNet,Ridge,Lasso
from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import SimpleExpSmoothing,Holt, ExponentialSmoothing
import warnings
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib


# Load all sheets into a dictionary of DataFrames
dfs = pd.read_excel(r"D:\360\Project - 1\Dataset\Coal Historical Prices _2020-24.xlsx", sheet_name=None)

# Print all sheet names to verify
print("Loaded Sheets:", dfs.keys())

# Access each sheet individually
df_2020 = dfs['2020']
df_2021 = dfs['2021']
df_2022 = dfs['2022']
df_2023 = dfs['2023']
df_2024 = dfs['2024']

# Display first few rows of one sheet (Example: 2020)
print(df_2020.head())
print(df_2021.head())
print(df_2022.head())
print(df_2023.head())
print(df_2024.head())

# Merge sheets, removing duplicate headers
df_combined = pd.concat([df if i == 0 else df.iloc[1:] for i, df in enumerate(dfs.values())], ignore_index=True)

# Save the cleaned data
df_combined.to_excel("Merged_Clean.xlsx", index=False)

#Loading Merged Dataset
df = pd.read_excel(r"D:\360\Project - 1\Dataset\Total_Merged_Coal_Historical_Prices.xlsx",skiprows=1)

# Assume your main DataFrame is `df` and 'Date' column exists
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
df.reset_index(drop=True, inplace=True)

# Extract time-based features
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['day_of_week'] = df['Date'].dt.dayofweek
df['quarter'] = df['Date'].dt.quarter


# Importing necessary module for creating a database engine
from sqlalchemy import create_engine
from urllib.parse import quote

# Setting up connection parameters for the MySQL database
user = 'root'  # Username
pw = quote('arunteja')  # Password
db = 'coal'  # Database name

# Creating an engine to connect to the MySQL database using SQLAlchemy
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# Renaming long column names to shorter ones (maximum identifier length of 64 characters)
df.columns = [
    "Date",
    "RB_4800kcal_fob",
    "RB_5500kcal_fob",
    "RB_5700kcal_fob",
    "RB_6000kcal_avg",
    "India_5500kcal_cfr",
    "year",
    "month",
    "day",
    "day_of_week",
    "quarter"
    
]

# Dumping the coal data into the MySQL database table named 'coal'
# 'if_exists' parameter is set to 'replace' to replace the table if it already exists
# 'chunksize' parameter is used to specify the number of rows to write at a time
# 'index' parameter is set to False to avoid writing DataFrame index as a column in the table
df.to_sql('coal', con=engine, if_exists='replace', chunksize=1000, index=False)

# loading data from database
# SQL query to select all records from the 'cancer' table in the MySQL database
sql = 'select * from coal'

# Reading data from the MySQL database table 'cancer' into a pandas DataFrame
df = pd.read_sql_query(sql, con=engine)

# Displaying the DataFrame
print(df)

# Initial Data Exploration
print("First 5 rows of the dataset:")
print(df.head())
print("\nLast 5 rows of the dataset:")
print(df.tail())
print("\nInfo (data types and non-null counts):")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())


# Exclude the date column from statistical calculations
df_numeric = df.select_dtypes(exclude=['datetime'])

stats_dict = {}

for column in df_numeric.columns:
    col_min = df_numeric[column].min()
    col_max = df_numeric[column].max()
    
    stats_dict[column] = {
        "Mean": df_numeric[column].mean(),
        "Median": df_numeric[column].median(),
        "Mode": df_numeric[column].mode()[0],
        "Variance": df_numeric[column].var(),
        "Standard Deviation": df_numeric[column].std(),
        "Skewness": df_numeric[column].skew(),
        "Kurtosis": df_numeric[column].kurtosis(),
        "Min": col_min,
        "Max": col_max,
        "Range": col_max - col_min
    }

# Convert the stats dictionary to a DataFrame for better readability
stats_df = pd.DataFrame(stats_dict).T

# Display the statistical summary
print("\nStatistical Summary (Mean, Median, Mode, Variance, Standard Deviation, Skewness, Kurtosis):")
print(stats_df)
stats_df.to_excel("coal_statistics_summary.xlsx", index=True)
print("Saved as coal_statistics_summary.xlsx")
df.columns

#Line Plot : To observe trends over time.
plt.figure(figsize=(12, 6))
for col in df.columns[1:]:  # Exclude Date column
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df[col], label=col, color='b')
    plt.xlabel("Date")
    plt.ylabel("Price (USD/t)")
    plt.title(f"Coal Price Trend: {col}")
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

#Histogram: To see the distribution of coal prices.
for col in df.columns[1:]:
    plt.figure(figsize=(10, 5))
    sns.histplot(df[col], kde=True, bins=30, color='g')
    plt.xlabel("Price (USD/t)")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {col}")
    plt.show()
    
#Boxplot : Box Plot: To identify outliers.
for col in df.columns[1:]:
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=df[col], color='r')
    plt.ylabel("Price (USD/t)")
    plt.title(f"Box Plot for {col}")
    plt.show()


#Heatmap : To find correlation between different coal prices.
plt.figure(figsize=(8, 5))
sns.heatmap(df.iloc[:, 1:].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Coal Prices")
plt.show()

#Scatter plot
for i in range(1, len(df.columns[1:])):  
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df[df.columns[i-1]], y=df[df.columns[i]], color='purple')
    plt.xlabel(df.columns[i-1])
    plt.ylabel(df.columns[i])
    plt.title(f"Scatter Plot: {df.columns[i-1]} vs {df.columns[i]}")
    plt.show()
    
#3d Scatter plot
for i in range(1, len(df.columns)-2):  
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df[df.columns[i]], df[df.columns[i+1]], df[df.columns[i+2]], c='r', marker='o')

    ax.set_xlabel(df.columns[i])
    ax.set_ylabel(df.columns[i+1])
    ax.set_zlabel(df.columns[i+2])
    ax.set_title(f"3D Scatter Plot: {df.columns[i]}, {df.columns[i+1]}, {df.columns[i+2]}")

    plt.show()

#Bubble Chart
for col in df.columns[1:]:
    plt.figure(figsize=(10, 5))
    plt.scatter(df["Date"], df[col], s=df[col]*0.5, alpha=0.5, color='orange')
    plt.xlabel("Date")
    plt.ylabel("Price (USD/t)")
    plt.title(f"Bubble Chart for {col}")
    plt.xticks(rotation=45)
    plt.show()

#Pair plot
sns.pairplot(df.iloc[:, 1:], height=3, aspect=1.5)  # Aspect increases width
plt.show()

#Time Series with Rolling Average (Smoother Trend)
for col in df.columns[1:]:  # Exclude Date column
    plt.figure(figsize=(12, 5))
    plt.plot(df['Date'], df[col], label="Actual", color='red', alpha=0.5)
    plt.plot(df['Date'], df[col].rolling(window=7).mean(), label="7-day Rolling Avg", color='blue', linewidth=2)
    plt.xlabel("Date")
    plt.ylabel("Price (USD/t)")
    plt.title(f"Time Series with Rolling Average: {col}")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

#Data Preprocessing

# Check for duplicates and drop them
print("\nChecking for duplicates:")
duplicates = df.duplicated().sum()
print(f"Total duplicate rows: {duplicates}")

# Drop duplicates if any
df.drop_duplicates(inplace=True)

# Check for missing values
print("\nChecking for missing values:")
missing_values = df.isnull().sum()
print(f"Missing values for each column:\n{missing_values}")

# Fill missing values with mean imputation
df.fillna(df.mean(), inplace=True)
print(f"Missing values after imputation:\n{df.isnull().sum()}")


#Time Series with Trend & Seasonality (Decomposition)
for col in df.columns[1:]:
    decomposition = seasonal_decompose(df[col], period=30, model='additive', extrapolate_trend='freq')

    plt.figure(figsize=(10, 8))

    plt.subplot(411)
    plt.plot(df['Date'], df[col], label="Original", color='orange')
    plt.legend()
    
    plt.subplot(412)
    plt.plot(df['Date'], decomposition.trend, label="Trend", color='blue')
    plt.legend()
    
    plt.subplot(413)
    plt.plot(df['Date'], decomposition.seasonal, label="Seasonality", color='green')
    plt.legend()
    
    plt.subplot(414)
    plt.plot(df['Date'], decomposition.resid, label="Residuals", color='red')
    plt.legend()


    plt.suptitle(f"Time Series Decomposition for {col}")
    plt.show()


# Plot ACF & PACF for each numerical column
for col in df.columns:
    if col != 'Date':  # Exclude date column
        plt.figure(figsize=(12, 5))

        # ACF Plot
        plt.subplot(1, 2, 1)
        plot_acf(df[col], lags=30, ax=plt.gca())
        plt.title(f"ACF Plot for {col}")

        # PACF Plot
        plt.subplot(1, 2, 2)
        plot_pacf(df[col], lags=30, ax=plt.gca(), method='yw')
        plt.title(f"PACF Plot for {col}")

        plt.tight_layout()
        plt.show()


# Creating lag plots for each numerical column 
for col in df.columns:
    if col != 'Date':  
        plt.figure(figsize=(6, 6))
        lag_plot(df[col], lag=1)
        plt.title(f"Lag Plot for {col} (Lag=1)")
        plt.show()



#Winsorization

# Define numerical columns for Winsorization
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Initialize Winsorizer for numerical columns using Gaussian method
winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=numerical_columns
)

# Apply Winsorization
df_winsorized = winsor.fit_transform(df)

# Compare boxplots before and after Winsorization
for column in numerical_columns:
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.boxplot(x=df[column], color='skyblue')
    plt.title(f'Original Boxplot - {column}')

    plt.subplot(1, 2, 2)
    sns.boxplot(x=df_winsorized[column], color='orange')
    plt.title(f'Winsorized Boxplot - {column}')

    plt.tight_layout()
    plt.show()

# Summary Statistics after Winsorization
print("\nSummary Statistics After Winsorization:")
print(df_winsorized.describe())

# Save the cleaned and winsorized DataFrame to Excel
df_winsorized.to_excel('cleaned_coal_price_data.xlsx', index=False)


# Initialize StandardScaler
#scaler = StandardScaler()
#scaler = MaxAbsScaler()
#scaler = RobustScaler()
scaler = MinMaxScaler()

# Select only numerical columns for scaling (exclude 'Date' or non-numeric columns)
numerical_columns = df_winsorized.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Apply StandardScaler
df_scaled = df_winsorized.copy()
df_scaled[numerical_columns] = scaler.fit_transform(df_winsorized[numerical_columns])

# Summary Statistics After Standardization
print("\nSummary Statistics After Standardization:")
print(df_scaled.describe())


#PIPELINE
# Define numerical columns
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numerical_columns = [col for col in numerical_columns if col != 'year' and col != 'month' and col != 'day' and col != 'day_of_week' and col != 'quarter']  # Optionally exclude time features

# Define the pipeline for numerical features
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),            # Step 1: Fill missing values with mean
    ('winsorizer', Winsorizer(capping_method='iqr',         # Step 2: Winsorization
                              tail='both', 
                              fold=1.5,
                              variables=None)),             # Apply to all numerical vars passed by ColumnTransformer
    ('scaler', MinMaxScaler())                              # Step 3: Scaling
])

# Create ColumnTransformer to apply pipeline only to numerical columns
preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, numerical_columns)
], remainder='passthrough')  # Keep other columns like 'Date'

# Apply the transformation
df_processed = preprocessor.fit_transform(df)

# Convert transformed output back to DataFrame
processed_column_names = numerical_columns + [col for col in df.columns if col not in numerical_columns]
df_processed = pd.DataFrame(df_processed, columns=processed_column_names)

# Ensure Date column is in datetime format again (if needed)
if 'Date' in df_processed.columns:
    df_processed['Date'] = pd.to_datetime(df_processed['Date'])

# Display
print("\n Data After Preprocessing Pipeline:")
print(df_processed.head())

joblib.dump(preprocessor, "preprocessing_pipeline.pkl")


target_col = 'RB_4800kcal_fob'
target_col = 'RB_5500kcal_fob'
target_col = 'RB_5700kcal_fob'
target_col = 'India_5500kcal_cfr'
target_col = 'RB_6000kcal_avg'


#DEEP LEARNING MODELS

#LSTM

# Drop 'Date' column

df_lstm = df_scaled.drop(columns=['Date']).copy()

n_lags = 7

# 1. Create lag features ONLY for the target variable
for lag in range(1, n_lags + 1):
    df_lstm[f'{target_col}_lag{lag}'] = df_lstm[target_col].shift(lag)

# 2. Drop rows with NaNs from lag creation
df_lagged = df_lstm.dropna().copy()

# 3. Prepare X and y
# Features: all columns *except* target (non-lagged) and including all lag features
X_cols = [col for col in df_lagged.columns if col != target_col]
X = df_lagged[X_cols].values
y = df_lagged[target_col].values

# Reshape X for LSTM: [samples, timesteps, features]
# In multivariate LSTM, timesteps = 1, features = number of features
X = X.reshape((X.shape[0], 1, X.shape[1]))

# 4. Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 5. Build and Train LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(128))
lstm_model.add(Dense(1))

lstm_model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=70, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5)

history = lstm_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

y_pred_lstm = lstm_model.predict(X_test)

# 7. Evaluation
mse = mean_squared_error(y_test, y_pred_lstm)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_lstm)
r2 = r2_score(y_test, y_pred_lstm)
mape = np.mean(np.abs((y_test - y_pred_lstm.flatten()) / y_test)) * 100

print("\nLSTM Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"R² Score: {r2:.4f}")#0.9138
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# 8. Plot Predictions
plt.figure(figsize=(10, 5))
plt.plot(y_test, label='Actual', linewidth=2)
plt.plot(y_pred_lstm, label='Predicted', linewidth=2)
plt.title('Multivariate LSTM Forecast vs Actual')
plt.xlabel('Time')
plt.ylabel(f'{target_col} (Scaled)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



#GRU
# Drop 'Date' column
df_gru = df_scaled.drop(columns=['Date']).copy()

n_lags = 7

# 1. Create lag features ONLY for the target variable
for lag in range(1, n_lags + 1):
    df_gru[f'{target_col}_lag{lag}'] = df_gru[target_col].shift(lag)

# 2. Drop rows with NaNs from lag creation
df_lagged = df_gru.dropna().copy()

# 3. Prepare X and y
X_cols = [col for col in df_lagged.columns if col != target_col]
X = df_lagged[X_cols].values
y = df_lagged[target_col].values

# Reshape X for GRU: [samples, timesteps, features]
X = X.reshape((X.shape[0], 1, X.shape[1]))

# 4. Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 5. Build and Train GRU model
gru_model = Sequential()
gru_model.add(GRU(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
gru_model.add(Dropout(0.2))
gru_model.add(GRU(128))
gru_model.add(Dense(1))

gru_model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=70, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5)

history = gru_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
# 6. Predictions
y_pred_gru = gru_model.predict(X_test)


# 7. Evaluation
mse = mean_squared_error(y_test, y_pred_gru)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_gru)
r2 = r2_score(y_test, y_pred_gru)
mape = np.mean(np.abs((y_test - y_pred_gru.flatten()) / y_test)) * 100

print("\nGRU Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# 8. Plot Predictions
plt.figure(figsize=(10, 5))
plt.plot(y_test, label='Actual', linewidth=2)
plt.plot(y_pred_gru, label='Predicted', linewidth=2)
plt.title('Multivariate GRU Forecast vs Actual')
plt.xlabel('Time')
plt.ylabel(f'{target_col} (Scaled)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#HYBRID MODELS

# Build the LSTM + GRU Hybrid Model (Improved)
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))
model.add(GRU(128, return_sequences=True))
model.add(Dropout(0.3))
model.add(GRU(64))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
from keras.optimizers import Adam
optimizer = Adam(learning_rate=0.001)  # Try also 0.0005 or 0.0001
model.compile(optimizer=optimizer, loss='mse')

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, verbose=1)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Make Predictions
y_pred_lstm_gru = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred_lstm_gru)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_lstm_gru)
r2 = r2_score(y_test, y_pred_lstm_gru)
mape = np.mean(np.abs((y_test - y_pred_lstm_gru.flatten()) / y_test)) * 100

print("\n LSTM + GRU Hybrid Model Evaluation Metrics:")
print(f"MSE  : {mse:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"MAE  : {mae:.6f}")
print(f"R²   : {r2:.4f}")
print(f"MAPE : {mape:.2f}%")

# Plot Predictions vs Actual
plt.figure(figsize=(10, 5))
plt.plot(y_test, label='Actual', linewidth=2)
plt.plot(y_pred_lstm_gru, label='Predicted', linewidth=2)
plt.title(' Hybrid LSTM + GRU Forecast vs Actual')
plt.xlabel('Time')
plt.ylabel(f'{target_col} (Scaled)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#ML MODELS

# Create lag features
# Start with Date and RB_6000kcal_avg
df_lagged = df_scaled.copy()

# Set Date as index
df_lagged.set_index('Date', inplace=True)

for lag in range(1, 8):  # 7 lag features
    df_lagged[f'lag_{lag}'] = df_lagged[f'{target_col}'].shift(lag)

# Drop NA after lagging
df_lagged.dropna(inplace=True)

# Features and target
X = df_lagged.drop(f'{target_col}', axis=1)
y = df_lagged[f'{target_col}']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

#Random Forest

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    max_features = 'sqrt',
    min_samples_leaf = 4,
    min_samples_split=10,
    random_state=42
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Test')
plt.plot(y_test.index, y_pred_rf, label='Forecast (Random Forest)', linestyle='--')
plt.legend()
plt.title(f"Random Forest Forecast - {target_col}")
plt.show()

# Metrics
mse = mean_squared_error(y_test, y_pred_rf)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_rf)
mape = np.mean(np.abs((y_test - y_pred_rf) / y_test)) * 100
r2 = r2_score(y_test, y_pred_rf)

print("Random Forest - Evaluation Metrics:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"R² Score: {r2:.4f}")


#Decision Tress
dt = DecisionTreeRegressor(max_depth=15, min_samples_leaf=2, min_samples_split = 2 ,random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Test')
plt.plot(y_test.index, y_pred_dt, label='Forecast (Decision Tree)', linestyle='--')
plt.legend()
plt.title(f"Decision Tree Forecast - {target_col}")
plt.show()

# Metrics
mse = mean_squared_error(y_test, y_pred_dt)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_dt)
mape = np.mean(np.abs((y_test - y_pred_dt) / y_test)) * 100
r2 = r2_score(y_test, y_pred_dt)

print("Decision Tree - Evaluation Metrics:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"R² Score: {r2:.4f}")#0.8705


#XgBoost

xgb = XGBRegressor(
    learning_rate=0.1,
    max_depth=3,
    n_estimators=100,
    random_state=42
)

xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Test')
plt.plot(y_test.index, y_pred_xgb, label='Forecast (XGBoost)', linestyle='--')
plt.legend()
plt.title(f"XGBoost Forecast - {target_col}")
plt.show()

# Metrics
mse = mean_squared_error(y_test, y_pred_xgb)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_xgb)
mape = np.mean(np.abs((y_test - y_pred_xgb) / y_test)) * 100
r2 = r2_score(y_test, y_pred_xgb)

print("XGBoost - Evaluation Metrics:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"R² Score: {r2:.4f}")#0.9179


#Linear Regression

# Assuming df_scaled and target_col are already defined
df_lagged = df_scaled.copy()

# Create lag features
n_lags = 7
for lag in range(1, n_lags + 1):
    df_lagged[f'lag_{lag}'] = df_lagged[f'{target_col}'].shift(lag)

df_lagged.dropna(inplace=True)

# Prepare features and target
X = df_lagged.drop(columns=[f'{target_col}', 'Date'])
y = df_lagged[f'{target_col}']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Show the features used for training
features_used = X.columns
print(features_used)

# Save the dataset with the selected features as CSV
X.to_csv('training_data_with_features.csv', index=False)


# Evaluation
mse = mean_squared_error(y_test, y_pred_lr)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_lr)
mape = np.mean(np.abs((y_test - y_pred_lr) / y_test)) * 100
r2 = r2_score(y_test, y_pred_lr)

print("Linear Regression - Evaluation Metrics:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"R² Score: {r2:.4f}")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred_lr, label='Predicted', linestyle='--')
plt.title(f'Linear Regression - Forecasting {target_col}')
plt.legend()
plt.show()

# === Save the model ===
model_filename = f"linear_regression_model_{target_col}_best_model.pkl"
joblib.dump(lr_model, model_filename)
print(f"Model saved as {model_filename}")


#Lasso
df_lagged = df_scaled.copy()

# Create lag features
n_lags = 7

for lag in range(1, n_lags + 1):
    df_lagged[f'lag_{lag}'] = df_lagged[f'{target_col}'].shift(lag)

df_lagged.dropna(inplace=True)

# Prepare features and target
X = df_lagged.drop(columns=[f'{target_col}', 'Date'])
y = df_lagged[f'{target_col}']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 3. Fit Lasso Regression
lasso_model = Lasso(alpha=0.001)
lasso_model.fit(X_train, y_train)

# 4. Predict
y_pred_lasso = lasso_model.predict(X_test)

# 5. Plot
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred_lasso, label='Forecast (Lasso)', linestyle='--')
plt.title(f'Lasso Regression Forecast - {target_col}')
plt.legend()
plt.show()

# 6. Evaluation Metrics
mse = mean_squared_error(y_test, y_pred_lasso)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_lasso)
mape = np.mean(np.abs((y_test - y_pred_lasso) / y_test)) * 100
r2 = r2_score(y_test, y_pred_lasso)

print("Lasso Regression - Evaluation Metrics:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"R² Score: {r2:.4f}")



#Elastic
df_lagged = df_scaled.copy()

# Create lag features
n_lags = 7

for lag in range(1, n_lags + 1):
    df_lagged[f'lag_{lag}'] = df_lagged[f'{target_col}'].shift(lag)

df_lagged.dropna(inplace=True)

# Prepare features and target
X = df_lagged.drop(columns=[f'{target_col}', 'Date'])
y = df_lagged[f'{target_col}']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ElasticNet model
elastic_model = ElasticNet(alpha=0.001, l1_ratio=0.9)
elastic_model.fit(X_train, y_train)
y_pred_elastic = elastic_model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred_elastic)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_elastic)
mape = np.mean(np.abs((y_test - y_pred_elastic) / y_test)) * 100
r2 = r2_score(y_test, y_pred_elastic)

print("ElasticNet Regression - Evaluation Metrics:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"R² Score: {r2:.4f}")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred_elastic, label='Predicted', linestyle='--')
plt.title(f'ElasticNet Regression - Forecasting {target_col}')
plt.legend()
plt.show()


# Ridge Regression
df_lagged = df_scaled.copy()

# Create lag features
n_lags = 7

for lag in range(1, n_lags + 1):
    df_lagged[f'lag_{lag}'] = df_lagged[f'{target_col}'].shift(lag)

df_lagged.dropna(inplace=True)

# Prepare features and target
X = df_lagged.drop(columns=[f'{target_col}', 'Date'])
y = df_lagged[f'{target_col}']


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Ridge Regression model
ridge_model = Ridge(alpha = 1.0)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred_ridge)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_ridge)
mape = np.mean(np.abs((y_test - y_pred_ridge) / y_test)) * 100
r2 = r2_score(y_test, y_pred_ridge)

print("Ridge Regression - Evaluation Metrics:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"R² Score: {r2:.4f}")

# Plot: Actual vs Predicted
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred_ridge, label='Predicted', linestyle='--')
plt.title(f'Ridge Regression - Forecasting {target_col}')
plt.xlabel('Date')
plt.legend()
plt.tight_layout()
plt.show()


#Support Vector Regression
df_lagged = df_scaled.copy()

# Create lag features
n_lags = 7

for lag in range(1, n_lags + 1):
    df_lagged[f'lag_{lag}'] = df_lagged[f'{target_col}'].shift(lag)

df_lagged.dropna(inplace=True)

# Prepare features and target
X = df_lagged.drop(columns=[f'{target_col}', 'Date'])
y = df_lagged[f'{target_col}']


# Store original index for plotting later
y_index = y.index

# Feature scaling (SVR is sensitive to scaling)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

# Train-test split
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_scaled, y_scaled, y_index, test_size=0.2, shuffle=False)

# SVR model
svr_model = SVR(C=100, epsilon=0.01, gamma='scale',kernel =  'linear')
svr_model.fit(X_train, y_train)
y_pred_scaled = svr_model.predict(X_test)

# Inverse transform predictions
y_pred_svr = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

# Evaluation
mse = mean_squared_error(y_test_actual, y_pred_svr)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual, y_pred_svr)
mape = np.mean(np.abs((y_test_actual - y_pred_svr) / y_test_actual)) * 100
r2 = r2_score(y_test_actual, y_pred_svr)

print("Support Vector Regression (SVR) - Evaluation Metrics:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"R² Score: {r2:.4f}")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(idx_test, y_test_actual, label='Actual')
plt.plot(idx_test, y_pred_svr, label='Predicted', linestyle='--')
plt.title(f'Support Vector Regression - Forecasting {target_col}')
plt.xlabel('Date')
plt.legend()
plt.tight_layout()
plt.show()


#TIME SERIES 
# Ignore warnings
warnings.filterwarnings("ignore")

# Prepare time series with Date as index
ts_df = df_winsorized[['Date', target_col]].copy()
ts_df['Date'] = pd.to_datetime(ts_df['Date'])
ts_df.set_index('Date', inplace=True)

# Set frequency and interpolate missing values
inferred_freq = pd.infer_freq(ts_df.index)
ts_df = ts_df.asfreq(inferred_freq if inferred_freq else 'MS')
ts_df[target_col].interpolate(inplace=True)

# Plot original time series
ts_df.plot(figsize=(10, 4), title=f"{target_col} Time Series")
plt.ylabel("Price")
plt.grid(True)
plt.show()

# Decompose the series
decomp = seasonal_decompose(ts_df[target_col], model='additive')
decomp.plot()
plt.show()

# ADF Test for stationarity
adf_result = adfuller(ts_df[target_col].dropna())
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
print("Stationary" if adf_result[1] < 0.05 else "Non-stationary: Consider differencing")

# First-order differencing (for stationarity)
ts_diff = ts_df[[target_col]].diff().dropna()
ts_diff.columns = [target_col]

# Plot differenced series
plt.figure(figsize=(10, 4))
plt.plot(ts_diff)
plt.title(f"First Order Differenced Series of {target_col}")
plt.grid(True)
plt.show()

# ADF Test after differencing
adf_result_diff = adfuller(ts_diff)
print("Differenced Series ADF Statistic:", adf_result_diff[0])
print("p-value:", adf_result_diff[1])
print("Stationary" if adf_result_diff[1] < 0.05 else "Still Non-Stationary")

# Train-Test Split
split_ratio = 0.8
split_index = int(len(ts_diff) * split_ratio)
y_train = ts_diff.iloc[:split_index][target_col]
y_test_arima = ts_diff.iloc[split_index:][target_col]

# Fit Auto ARIMA to find best (p,d,q)
auto_model = auto_arima(y_train,
                        seasonal=False,
                        stepwise=True,
                        trace=True,
                        error_action='ignore',
                        suppress_warnings=True)

print("\nBest ARIMA order from Auto ARIMA:", auto_model.order)

# Fit ARIMA with best order
best_model = ARIMA(y_train, order=auto_model.order)
best_model_fit = best_model.fit()
print(best_model_fit.summary())

# Forecast
y_pred_arima = best_model_fit.forecast(steps=len(y_test_arima))
y_pred_arima = pd.Series(y_pred_arima, index=y_test_arima.index)

# Plot
plt.figure(figsize=(12, 5))
plt.plot(y_train, label='Train')
plt.plot(y_test_arima, label='Test')
plt.plot(y_pred_arima, label='Auto ARIMA Forecast')
plt.title(f'Auto ARIMA Forecast for {target_col} (Differenced)')
plt.xlabel('Date')
plt.ylabel('Differenced Price')
plt.legend()
plt.grid(True)
plt.show()

# Evaluation Metrics
non_zero_mask = y_test_arima != 0
mse = mean_squared_error(y_test_arima, y_pred_arima)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_arima, y_pred_arima)
mape = np.mean(np.abs((y_test_arima[non_zero_mask] - y_pred_arima[non_zero_mask]) / y_test_arima[non_zero_mask])) * 100
r2 = r2_score(y_test_arima, y_pred_arima)

print("\nAuto ARIMA Evaluation Metrics:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"R² Score: {r2:.4f}")


#SARIMA
seasonal_period = 12
# Train-Test Split (same as ARIMA)
split_ratio = 0.8
split_index = int(len(ts_diff) * split_ratio)
y_train = ts_diff.iloc[:split_index][target_col]
y_test_sarima = ts_diff.iloc[split_index:][target_col]

# Auto SARIMA
auto_sarima_model = auto_arima(y_train,
                                seasonal=True,
                                m=seasonal_period,
                                stepwise=True,
                                trace=True,
                                suppress_warnings=True,
                                error_action="ignore")

print("\nBest SARIMA order from Auto SARIMA:")
print(f"Non-seasonal order: {auto_sarima_model.order}")
print(f"Seasonal order: {auto_sarima_model.seasonal_order}")

# Fit SARIMA with best parameters
sarima_model = SARIMAX(
    y_train,
    order=auto_sarima_model.order,
    seasonal_order=auto_sarima_model.seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarima_fit = sarima_model.fit(disp=False)

# Forecast
y_pred_sarima = sarima_fit.forecast(steps=len(y_test_sarima))
y_pred_sarima = pd.Series(y_pred_sarima, index=y_test_sarima.index)

# Plot forecast vs actual
plt.figure(figsize=(12, 5))
plt.plot(y_train, label='Train')
plt.plot(y_test_sarima, label='Test')
plt.plot(y_pred_sarima, label='Auto SARIMA Forecast')
plt.title(f'Auto SARIMA Forecast for {target_col}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Evaluation Metrics
non_zero_mask = y_test_sarima != 0
mse = mean_squared_error(y_test_sarima, y_pred_sarima)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_sarima, y_pred_sarima)
mape = np.mean(np.abs((y_test_sarima[non_zero_mask] - y_pred_sarima[non_zero_mask]) / y_test_sarima[non_zero_mask])) * 100
r2 = r2_score(y_test_sarima, y_pred_sarima)

print("\nAuto SARIMA Evaluation Metrics:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"R² Score: {r2:.4f}")



#Simple Exponential Smoothing (SES)
# Train-test split
split_index = int(len(ts_diff) * 0.8)
y_train = ts_diff.iloc[:split_index][target_col]
y_test_ses = ts_diff.iloc[split_index:][target_col]

# Fit model
ses_model = SimpleExpSmoothing(y_train).fit()
y_pred_ses = ses_model.forecast(len(y_test_ses))
y_pred_ses = pd.Series(y_pred_ses, index=y_test_ses.index)

# Plot
plt.figure(figsize=(12, 5))
plt.plot(y_train, label='Train')
plt.plot(y_test_ses, label='Test')
plt.plot(y_pred_ses, label='SES Forecast')
plt.title(f'Simple Exponential Smoothing Forecast for {target_col}')
plt.legend()
plt.grid(True)
plt.show()

# Evaluation
non_zero_mask = y_test_ses != 0
mse = mean_squared_error(y_test_ses, y_pred_ses)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_ses, y_pred_ses)
mape = np.mean(np.abs((y_test_ses[non_zero_mask] - y_pred_ses[non_zero_mask]) / y_test_ses[non_zero_mask])) * 100
r2 = r2_score(y_test_ses, y_pred_ses)

print("\nSES Evaluation Metrics:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"R² Score: {r2:.4f}")


#Holt’s Linear Trend
y_test_holt = ts_diff.iloc[split_index:][target_col]

# Fit model
holt_model = Holt(y_train).fit()
y_pred_holt = holt_model.forecast(len(y_test_holt))
y_pred_holt = pd.Series(y_pred_holt, index=y_test_holt.index)

# Plot
plt.figure(figsize=(12, 5))
plt.plot(y_train, label='Train')
plt.plot(y_test_holt, label='Test')
plt.plot(y_pred_holt, label='Holt’s Forecast')
plt.title(f'Holt’s Linear Trend Forecast for {target_col}')
plt.legend()
plt.grid(True)
plt.show()

# Evaluation
non_zero_mask = y_test_holt != 0
mse = mean_squared_error(y_test_holt, y_pred_holt)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_holt, y_pred_holt)
mape = np.mean(np.abs((y_test_holt[non_zero_mask] - y_pred_holt[non_zero_mask]) / y_test_holt[non_zero_mask])) * 100
r2 = r2_score(y_test_holt, y_pred_holt)

print("\nHolt’s Evaluation Metrics:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"R² Score: {r2:.4f}")


# Holt-Winters Additive 
y_test_hw = ts_diff.iloc[split_index:][target_col]

# Fit model
hw_model = ExponentialSmoothing(y_train, trend='add', seasonal='add', seasonal_periods=12).fit()
y_pred_hw = hw_model.forecast(len(y_test_hw))
y_pred_hw = pd.Series(y_pred_hw, index=y_test_hw.index)

# Plot
plt.figure(figsize=(12, 5))
plt.plot(y_train, label='Train')
plt.plot(y_test_hw, label='Test')
plt.plot(y_pred_hw, label='Holt-Winters Forecast')
plt.title(f'Holt-Winters Additive Forecast for {target_col}')
plt.legend()
plt.grid(True)
plt.show()

# Evaluation
non_zero_mask = y_test_hw != 0
mse = mean_squared_error(y_test_hw, y_pred_hw)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_hw, y_pred_hw)
mape = np.mean(np.abs((y_test_hw[non_zero_mask] - y_pred_hw[non_zero_mask]) / y_test_hw[non_zero_mask])) * 100
r2 = r2_score(y_test_hw, y_pred_hw)

print("\nHolt-Winters Evaluation Metrics:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"R² Score: {r2:.4f}")




#Model Evaluation 

# Optional: Safe MAPE function
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

# Store metrics for all models
metrics_list = []

#    Regression Models 

# Linear Regression
metrics_list.append({
    'Model': 'Linear Regression',
    'MSE': mean_squared_error(y_test, y_pred_lr),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
    'MAE': mean_absolute_error(y_test, y_pred_lr),
    'MAPE': mean_absolute_percentage_error(y_test, y_pred_lr),
    'R2': r2_score(y_test, y_pred_lr)
})

# Lasso Regression
metrics_list.append({
    'Model': 'Lasso Regression',
    'MSE': mean_squared_error(y_test, y_pred_lasso),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lasso)),
    'MAE': mean_absolute_error(y_test, y_pred_lasso),
    'MAPE': mean_absolute_percentage_error(y_test, y_pred_lasso),
    'R2': r2_score(y_test, y_pred_lasso)
})

# Ridge Regression
metrics_list.append({
    'Model': 'Ridge Regression',
    'MSE': mean_squared_error(y_test, y_pred_ridge),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
    'MAE': mean_absolute_error(y_test, y_pred_ridge),
    'MAPE': mean_absolute_percentage_error(y_test, y_pred_ridge),
    'R2': r2_score(y_test, y_pred_ridge)
})

# ElasticNet Regression
metrics_list.append({
    'Model': 'ElasticNet Regression',
    'MSE': mean_squared_error(y_test, y_pred_elastic),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_elastic)),
    'MAE': mean_absolute_error(y_test, y_pred_elastic),
    'MAPE': mean_absolute_percentage_error(y_test, y_pred_elastic),
    'R2': r2_score(y_test, y_pred_elastic)
})

# SVR
metrics_list.append({
    'Model': 'SVR',
    'MSE': mean_squared_error(y_test_actual, y_pred_svr),
    'RMSE': np.sqrt(mean_squared_error(y_test_actual, y_pred_svr)),
    'MAE': mean_absolute_error(y_test_actual, y_pred_svr),
    'MAPE': mean_absolute_percentage_error(y_test_actual, y_pred_svr),
    'R2': r2_score(y_test_actual, y_pred_svr)
})

# Decision Tree
metrics_list.append({
    'Model': 'Decision Tree',
    'MSE': mean_squared_error(y_test, y_pred_dt),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_dt)),
    'MAE': mean_absolute_error(y_test, y_pred_dt),
    'MAPE': mean_absolute_percentage_error(y_test, y_pred_dt),
    'R2': r2_score(y_test, y_pred_dt)
})

# Random Forest
metrics_list.append({
    'Model': 'Random Forest',
    'MSE': mean_squared_error(y_test, y_pred_rf),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
    'MAE': mean_absolute_error(y_test, y_pred_rf),
    'MAPE': mean_absolute_percentage_error(y_test, y_pred_rf),
    'R2': r2_score(y_test, y_pred_rf)
})

# XGBoost
metrics_list.append({
    'Model': 'XGBoost',
    'MSE': mean_squared_error(y_test, y_pred_xgb),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
    'MAE': mean_absolute_error(y_test, y_pred_xgb),
    'MAPE': mean_absolute_percentage_error(y_test, y_pred_xgb),
    'R2': r2_score(y_test, y_pred_xgb)
})

#     Deep Learning Models 

# LSTM
metrics_list.append({
    'Model': 'LSTM',
    'MSE': mean_squared_error(y_test, y_pred_lstm),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lstm)),
    'MAE': mean_absolute_error(y_test, y_pred_lstm),
    'MAPE': mean_absolute_percentage_error(y_test, y_pred_lstm.flatten()),
    'R2': r2_score(y_test, y_pred_lstm)
})

# GRU
metrics_list.append({
    'Model': 'GRU',
    'MSE': mean_squared_error(y_test, y_pred_gru),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_gru)),
    'MAE': mean_absolute_error(y_test, y_pred_gru),
    'MAPE': mean_absolute_percentage_error(y_test, y_pred_gru.flatten()),
    'R2': r2_score(y_test, y_pred_gru)
})

# LSTM + GRU Hybrid
metrics_list.append({
    'Model': 'LSTM + GRU Hybrid',
    'MSE': mean_squared_error(y_test, y_pred_lstm_gru),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lstm_gru)),
    'MAE': mean_absolute_error(y_test, y_pred_lstm_gru),
    'MAPE': mean_absolute_percentage_error(y_test, y_pred_lstm_gru.flatten()),
    'R2': r2_score(y_test, y_pred_lstm_gru)
})

#   Statistical Time Series Models 

# ARIMA
metrics_list.append({
    'Model': 'ARIMA',
    'MSE': mean_squared_error(y_test_arima, y_pred_arima),
    'RMSE': np.sqrt(mean_squared_error(y_test_arima, y_pred_arima)),
    'MAE': mean_absolute_error(y_test_arima, y_pred_arima),
    'MAPE': mean_absolute_percentage_error(y_test_arima, y_pred_arima),
    'R2': r2_score(y_test_arima, y_pred_arima)
})

# SARIMA
metrics_list.append({
    'Model': 'SARIMA',
    'MSE': mean_squared_error(y_test_sarima, y_pred_sarima),
    'RMSE': np.sqrt(mean_squared_error(y_test_sarima, y_pred_sarima)),
    'MAE': mean_absolute_error(y_test_sarima, y_pred_sarima),
    'MAPE': mean_absolute_percentage_error(y_test_sarima, y_pred_sarima),
    'R2': r2_score(y_test_sarima, y_pred_sarima)
})

# SES
metrics_list.append({
    'Model': 'SES',
    'MSE': mean_squared_error(y_test_ses, y_pred_ses),
    'RMSE': np.sqrt(mean_squared_error(y_test_ses, y_pred_ses)),
    'MAE': mean_absolute_error(y_test_ses, y_pred_ses),
    'MAPE': mean_absolute_percentage_error(y_test_ses, y_pred_ses),
    'R2': r2_score(y_test_ses, y_pred_ses)
})

# Holt’s Linear Trend
metrics_list.append({
    'Model': "Holt's Linear Trend",
    'MSE': mean_squared_error(y_test_holt, y_pred_holt),
    'RMSE': np.sqrt(mean_squared_error(y_test_holt, y_pred_holt)),
    'MAE': mean_absolute_error(y_test_holt, y_pred_holt),
    'MAPE': mean_absolute_percentage_error(y_test_holt, y_pred_holt),
    'R2': r2_score(y_test_holt, y_pred_holt)
})

# Holt-Winters Additive
metrics_list.append({
    'Model': 'Holt-Winters Additive',
    'MSE': mean_squared_error(y_test_hw, y_pred_hw),
    'RMSE': np.sqrt(mean_squared_error(y_test_hw, y_pred_hw)),
    'MAE': mean_absolute_error(y_test_hw, y_pred_hw),
    'MAPE': mean_absolute_percentage_error(y_test_hw, y_pred_hw),
    'R2': r2_score(y_test_hw, y_pred_hw)
})


# Results Display 

# Convert to DataFrame and sort by R²
eval_df = pd.DataFrame(metrics_list)
eval_df = eval_df.sort_values(by='R2', ascending=False).reset_index(drop=True)

# Display the comparison table
print("\n  Updated Model Evaluation Comparison:")
print(eval_df)

# Show best model
best_model = eval_df.loc[0]
best_model_name = best_model['Model']
print(f"\n  Best Model Based on R²: {best_model['Model']} (R² = {best_model['R2']:.4f})")

# Dynamically generate model variable name
model_var_name = best_model_name.lower().replace(' ', '_').replace('+', '').replace("’", '').replace("'", '').replace('-', '') + '_model'

# Get the best model object dynamically
best_model_object = globals().get(model_var_name)

# Assume target_column is defined somewhere in your loop or script
target_column = target_col.lower().replace(' ', '_')

# Save the model based on type
if 'lstm' in model_var_name or 'gru' in model_var_name:  # lowercase check
    filename = f"{model_var_name}_{target_col}_best_model.h5"
    best_model_object.save(filename)
    print(f" Saved Keras model: {filename}")
else:
    filename = f"{model_var_name}_{target_col}_best_model.pkl"
    joblib.dump(best_model_object, filename)
    print(f" Saved model: {filename}")

'''
 1)  Updated Model Evaluation Comparison - RB_4800kcal_fob
                    Model         MSE  ...        MAPE        R2
0           Random Forest    0.000126  ...    2.792455  0.946365
1                     GRU    0.000131  ...    2.706756  0.944305
2                    LSTM    0.000143  ...    2.879137  0.939126
3        Lasso Regression    0.000181  ...    3.006622  0.923229
4   ElasticNet Regression    0.000184  ...    3.045082  0.922022
5                     SVR    0.000190  ...    2.354225  0.919378
6       Linear Regression    0.000196  ...    3.608820  0.916861
7                 XGBoost    0.000199  ...    3.706419  0.915538
8       LSTM + GRU Hybrid    0.000202  ...    3.311095  0.914069
9        Ridge Regression    0.000214  ...    3.860698  0.909228
10          Decision Tree    0.000557  ...    5.159543  0.763684
11                 SARIMA   97.945340  ...  100.000000 -0.000044
12                  ARIMA   99.272570  ...   94.486569 -0.013596
13                    SES  114.058805  ...  120.679988 -0.164567
14  Holt-Winters Additive  183.377363  ...  178.488817 -0.872325
15    Holt's Linear Trend  223.322376  ...  123.176947 -1.280173

[16 rows x 6 columns]

  Best Model Based on R²: Random Forest (R² = 0.9464)
  
'''

'''

 2)  Updated Model Evaluation Comparison - RB_5500kcal_fob
                    Model         MSE  ...        MAPE        R2
0                     GRU    0.000022  ...    1.093853  0.990986
1                     SVR    0.000032  ...    1.424814  0.987134
2                    LSTM    0.000034  ...    1.333724  0.986167
3       Linear Regression    0.000048  ...    1.727353  0.980412
4        Ridge Regression    0.000060  ...    1.975085  0.975659
5           Random Forest    0.000093  ...    2.306959  0.962210
6                 XGBoost    0.000175  ...    3.430729  0.928785
7   ElasticNet Regression    0.000187  ...    3.850822  0.924308
8        Lasso Regression    0.000200  ...    4.027509  0.918844
9       LSTM + GRU Hybrid    0.000231  ...    3.384296  0.906428
10          Decision Tree    0.000324  ...    4.165842  0.868457
11                 SARIMA  173.979962  ...   99.196417  0.004535
12                  ARIMA  176.084230  ...   94.517938 -0.007505
13                    SES  203.713590  ...  121.388771 -0.165592
14  Holt-Winters Additive  245.647225  ...  167.170068 -0.405525
15    Holt's Linear Trend  459.111113  ...  151.716788 -1.626905


[16 rows x 6 columns]

  Best Model Based on R²: GRU (R² = 0.9910)
'''


'''

 3)  Updated Model Evaluation Comparison - 5700kcal
                    Model         MSE  ...        MAPE        R2
0       Linear Regression    0.000059  ...    1.770256  0.977213
1                     SVR    0.000063  ...    1.759392  0.975601
2                     GRU    0.000074  ...    1.760936  0.971426
3                    LSTM    0.000088  ...    1.964482  0.965960
4       LSTM + GRU Hybrid    0.000091  ...    1.996473  0.965044
5        Ridge Regression    0.000154  ...    2.715956  0.940659
6           Random Forest    0.000178  ...    3.893291  0.931333
7                 XGBoost    0.000239  ...    4.588762  0.907779
8           Decision Tree    0.000313  ...    4.844345  0.879355
9   ElasticNet Regression    0.000456  ...    7.235948  0.824022
10       Lasso Regression    0.000525  ...    7.835274  0.797518
11  Holt-Winters Additive  297.491972  ...  137.063226  0.057157
12                 SARIMA  313.754478  ...   98.051590  0.005616
13                  ARIMA  316.947023  ...   92.861271 -0.004502
14                    SES  334.192908  ...  119.581025 -0.059159
15    Holt's Linear Trend  428.307058  ...  109.559669 -0.357436

[16 rows x 6 columns]

  Best Model Based on R²: Linear Regression (R² = 0.9772)
'''

'''
  4) Updated Model Evaluation Comparison - india_550kcal_cfr
                    Model         MSE  ...        MAPE        R2
0                     GRU    0.000091  ...    2.279649  0.927190
1                    LSTM    0.000117  ...    2.718341  0.906544
2                     SVR    0.000127  ...    1.976906  0.898765
3        Lasso Regression    0.000130  ...    2.641427  0.896216
4   ElasticNet Regression    0.000132  ...    2.709805  0.894155
5           Random Forest    0.000165  ...    3.162409  0.868364
6        Ridge Regression    0.000220  ...    3.566032  0.823899
7       Linear Regression    0.000252  ...    3.833286  0.798566
8       LSTM + GRU Hybrid    0.000307  ...    4.228143  0.754441
9           Decision Tree    0.000463  ...    4.878860  0.629928
10                XGBoost    0.000525  ...    5.865954  0.580134
11                  ARIMA  123.091697  ...   98.991174 -0.036458
12                 SARIMA  123.469362  ...  102.138082 -0.039638
13    Holt's Linear Trend  137.377559  ...  109.935362 -0.156748
14                    SES  177.828934  ...  119.167999 -0.497357
15  Holt-Winters Additive  240.619545  ...  143.446182 -1.026067

[16 rows x 6 columns]

  Best Model Based on R²: GRU (R² = 0.9272)
'''

'''

  5) Updated Model Evaluation Comparison - 60000kcal
                    Model         MSE  ...        MAPE        R2
0                    LSTM    0.000167  ...    2.751735  0.935167
1                     GRU    0.000179  ...    2.471570  0.930432
2        Ridge Regression    0.000193  ...    2.972018  0.924782
3       LSTM + GRU Hybrid    0.000210  ...    3.387631  0.918245
4                 XGBoost    0.000211  ...    2.537611  0.917937
5           Random Forest    0.000244  ...    3.307786  0.905051
6       Linear Regression    0.000268  ...    2.397441  0.895664
7                     SVR    0.000282  ...    2.019652  0.890129
8   ElasticNet Regression    0.000287  ...    4.045085  0.888536
9        Lasso Regression    0.000296  ...    3.670050  0.884777
10          Decision Tree    0.000333  ...    4.301627  0.870545
11                  ARIMA  580.173256  ...  101.554488 -0.039750
12                 SARIMA  581.260924  ...  105.748352 -0.041700
13    Holt's Linear Trend  605.637659  ...  113.816870 -0.085386
14                    SES  681.562057  ...  161.642941 -0.221453
15  Holt-Winters Additive  763.365285  ...  228.040214 -0.368056

[16 rows x 6 columns]

  Best Model Based on R²: LSTM (R² = 0.9352)
'''
