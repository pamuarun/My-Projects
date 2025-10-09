# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 12:30:29 2025

@author: Arun Teja
"""

# ======================
# Import Libraries
# ======================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ======================
# Load Dataset
# ======================
df = pd.read_csv(r"D:\360\Project  - 3\Datasets\Main Dataset\Project_dataset_updated.csv")


# Importing necessary module for creating a database engine
from sqlalchemy import create_engine
from urllib.parse import quote

# Setting up connection parameters for the MySQL database
user = 'root'  # Username
pw = quote('arunteja')  # Password
db = 'risk'  # Database name

# Creating an engine to connect to the MySQL database using SQLAlchemy
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# Dumping the coal data into the MySQL database table named 'risk'
# 'if_exists' parameter is set to 'replace' to replace the table if it already exists
# 'chunksize' parameter is used to specify the number of rows to write at a time
# 'index' parameter is set to False to avoid writing DataFrame index as a column in the table
df.to_sql('risk', con=engine, if_exists='replace', chunksize=1000, index=False)

# loading data from database
# SQL query to select all records from the 'cancer' table in the MySQL database
sql = 'select * from risk'

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


# Separate numerical & categorical columns
num_df = df.select_dtypes(include=['int64', 'float64'])
cat_df = df.select_dtypes(include=['object', 'bool'])

# Numerical Descriptive Statistics
num_stats = pd.DataFrame({
    "Mean": num_df.mean(),
    "Median": num_df.median(),
    "Mode": num_df.mode().iloc[0],   # First mode if multiple
    "Variance": num_df.var(),
    "Std_Deviation": num_df.std(),
    "Skewness": num_df.skew(),
    "Kurtosis": num_df.kurt()
}).round(3)

print(" Numerical Descriptive Statistics")
print(num_stats, "\n")

# Categorical Descriptive Statistics 
cat_stats = pd.DataFrame({
    "Mode": cat_df.mode().iloc[0],
    "Unique_Count": cat_df.nunique(),
    "Most_Frequent_Value_Count": cat_df.apply(lambda x: x.value_counts().iloc[0])
})

print(" Categorical Descriptive Statistics")
print(cat_stats, "\n")

#  Save results
#num_stats.to_csv("numerical_statistics.csv")
#cat_stats.to_csv("categorical_statistics.csv")


# Set plot style
sns.set(style="whitegrid")


# 1. UNIVARIATE ANALYSIS
# Numerical columns: Histogram + KDE
num_cols = df.select_dtypes(include=['int64','float64']).columns

for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True, bins=30, color="skyblue")
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()

# Categorical columns: Countplot
cat_cols = df.select_dtypes(include=['object','bool']).columns

for col in cat_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(x=df[col], palette="Set2")
    plt.title(f"Count of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()


# 2. BIVARIATE ANALYSIS
# Scatterplots for key numerical relationships
plt.figure(figsize=(6,4))
sns.scatterplot(x="vehicle_count_per_hr", y="avg_speed_kmph", hue="accident_occurred", data=df, palette="coolwarm")
plt.title("Vehicle Count vs Avg Speed (Accidents Highlighted)")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x="weather", y="avg_speed_kmph", data=df)
plt.title("Avg Speed vs Weather")
plt.xticks(rotation=30)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (Numerical Features)")
plt.show()


# 3. MULTIVARIATE ANALYSIS
# Pairplot of key traffic features
sns.pairplot(df[["vehicle_count_per_hr", "avg_speed_kmph", "violations_count", "accident_occurred"]], 
             hue="accident_occurred", diag_kind="kde", palette="husl")
plt.show()

# FacetGrid: Speed vs Vehicle Count split by Peak Hours
g = sns.FacetGrid(df, col="is_peak", hue="accident_occurred", height=4, palette="coolwarm")
g.map(sns.scatterplot, "vehicle_count_per_hr", "avg_speed_kmph").add_legend()
plt.show()

# Stacked Bar: Road Type vs Accident Occurrence
road_acc = pd.crosstab(df["road_type"], df["accident_occurred"])
road_acc.plot(kind="bar", stacked=True, figsize=(6,4), colormap="viridis")
plt.title("Accidents by Road Type")
plt.ylabel("Count")
plt.show()
