# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 09:06:56 2025

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
# Save the DataFrame to an Excel file


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

# Calculate statistical measures
stats_dict = {}

for column in df_numeric.columns:
    stats_dict[column] = {
        "Mean": df_numeric[column].mean(),
        "Median": df_numeric[column].median(),
        "Mode": df_numeric[column].mode()[0],
        "Variance": df_numeric[column].var(),
        "Standard Deviation": df_numeric[column].std(),
        "Skewness": df_numeric[column].skew(),
        "Kurtosis": df_numeric[column].kurtosis()
    }

# Convert the stats dictionary to a DataFrame for better readability
stats_df = pd.DataFrame(stats_dict).T

# Display the statistical summary
print("\nStatistical Summary (Mean, Median, Mode, Variance, Standard Deviation, Skewness, Kurtosis):")
print(stats_df)

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
