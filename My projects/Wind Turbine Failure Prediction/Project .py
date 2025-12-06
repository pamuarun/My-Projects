# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 12:47:38 2025

@author: Arun Teja
"""
#Loading All Libraries
# Importing libraries for data manipulation and analysis
import pandas as pd  # For handling and manipulating data in DataFrames
import numpy as np  # For numerical computations

# Importing libraries for data visualization
import matplotlib.pyplot as plt  # For creating static, animated, and interactive visualizations
import seaborn as sns  # For advanced statistical visualizations

# Importing library for Principal Component Analysis (PCA)
from sklearn.decomposition import PCA  # For dimensionality reduction and feature extraction

# Importing library for outlier handling
from feature_engine.outliers import Winsorizer  # For outlier treatment by capping extreme values

# Importing library for clustering
from sklearn.cluster import KMeans  # For clustering data into groups

# Importing function for lag plot visualization
from pandas.plotting import lag_plot  # For visualizing time series data correlation with its lag

# Importing library for statistical tests on time series data
from statsmodels.tsa.stattools import adfuller  # For performing the Augmented Dickey-Fuller test (stationarity check)

# Importing library for statistical hypothesis testing
from scipy.stats import ttest_ind  # For conducting independent t-tests between groups

# Importing library for data standardization
from sklearn.preprocessing import StandardScaler  # For scaling features to have zero mean and unit variance


# Load the dataset
df = pd.read_csv(r"D:\360\Mini Project\Dataset\Wind_turbine.csv")

# Display first 5 rows of the dataset
print("Head of the dataset:")
print(df.head())

# Display the shape of the dataset
print("\nShape of the dataset:")
print(df.shape)

# Display column datatypes
print("\nColumn data types:")
print(df.dtypes)

# Display a concise summary of the DataFrame
print("\nDataset Info:")
df.info()

# Check for null values
print("\nNull values in each column:")
print(df.isnull().sum())

# Get the number of non-null values in each column
print("\nNumber of non-null values in each column:")
print(df.count())

# Displaying summary statistics of the DataFrame with a tag for description
print("# Description of Dataset Summary Statistics:")
print(df.describe())

# Impute missing values in float columns with their mean
for column in df.select_dtypes(include=['float64']).columns:
    df[column].fillna(df[column].mean(), inplace=True)

# Check for remaining null values
print("Null values after imputation:")
print(df.isnull().sum())

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

# Drop duplicates if any
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"Duplicates dropped. New shape: {df.shape}")

# Define numerical columns for Winsorization
numerical_columns = df.select_dtypes(include=['float64']).columns.tolist()

# Initialize Winsorizer for numerical columns using IQR method
winsor = Winsorizer(
    capping_method='iqr',  # Use IQR for capping
    tail='both',           # Cap both tails (lower and upper outliers)
    fold=1.5,              # Multiplier for the IQR (1.5 is standard for moderate outliers)
    variables=numerical_columns  # Apply only to numerical columns
)

# Apply Winsorization to the dataset (only numerical columns will be transformed)
df_winsorized = winsor.fit_transform(df)

# Compare boxplots before and after Winsorization
for column in numerical_columns:
    plt.figure(figsize=(12, 6))
    
    # Original data boxplot
    plt.subplot(1, 2, 1)
    sns.boxplot(x=df[column], color='skyblue')
    plt.title(f'Original Boxplot - {column}')
    
    # Winsorized data boxplot
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df_winsorized[column], color='orange')
    plt.title(f'Winsorized Boxplot - {column}')
    
    plt.tight_layout()
    plt.show()

# Display summary statistics after Winsorization
print("\nSummary Statistics After Winsorization:")
print(df_winsorized.describe())

# Statistical metrics for each column with range, max, and min added
print("\nStatistical metrics for each column (including range, max, and min):")
for column in df_winsorized.columns:
    if df_winsorized[column].dtype in ['int64', 'float64']:
        column_range = df_winsorized[column].max() - df_winsorized[column].min()  # Calculate range
        print(f"\nColumn: {column}")
        print(f"Mean: {df_winsorized[column].mean()}")
        print(f"Median: {df_winsorized[column].median()}")
        print(f"Mode: {df_winsorized[column].mode()[0]}")  # Display the first mode
        print(f"Variance: {df_winsorized[column].var()}")
        print(f"Standard Deviation: {df_winsorized[column].std()}")
        print(f"Skewness: {df_winsorized[column].skew()}")
        print(f"Kurtosis: {df_winsorized[column].kurt()}")
        print(f"Range: {column_range}")
        print(f"Max: {df_winsorized[column].max()}")
        print(f"Min: {df_winsorized[column].min()}")
    else:
        print(f"\nColumn: {column} is not numeric, skipping...")

#Graphical Analysis

# Histogram
for column in numerical_columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df_winsorized[column], kde=True, bins=30, color='blue')
    plt.title(f"Histogram for {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

#Boxplot
for column in numerical_columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df_winsorized[column], color='orange')
    plt.title(f"Boxplot for {column}")
    plt.xlabel(column)
    plt.show()

#Time Plot
if "date" in df_winsorized.columns:
    time_column = "date"
    for column in numerical_columns:
        plt.figure(figsize=(12, 6))
        plt.plot(df_winsorized[time_column], df_winsorized[column], label=column, color='blue')
        plt.title(f"date Plot for {column}")
        plt.xlabel("date")
        plt.ylabel(column)
        plt.legend()
        plt.show()

#Lag Plot (For each numerical column)
for column in numerical_columns:
    plt.figure(figsize=(8, 6))
    lag_plot(df_winsorized[column])
    plt.title(f"Lag Plot for {column}")
    plt.show()

#Scatter Plot (Pairwise scatter plots)
for i, col1 in enumerate(numerical_columns):
    for col2 in numerical_columns[i+1:]:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df_winsorized[col1], y=df_winsorized[col2])
        plt.title(f"Scatter Plot: {col1} vs {col2}")
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.show()

#Stationarity Check
for column in numerical_columns:
    result = adfuller(df_winsorized[column])
    print(f"\nADF Test Statistic for {column}: {result[0]}")
    print(f"p-value: {result[1]}")
    print("Stationary" if result[1] < 0.05 else "Non-stationary")

#Random Walk 

# Iterate over each numerical column in the dataset
for column in df_winsorized.select_dtypes(include=['float64']).columns:
    # Create a random walk using the column's data as a base
    random_walk = np.cumsum(np.random.randn(len(df_winsorized[column]))) + df_winsorized[column].iloc[0]
    
    # Plot the random walk for the current column
    plt.figure(figsize=(10, 6))
    plt.plot(random_walk, color='purple')
    plt.title(f"Random Walk for {column}")
    plt.xlabel("Steps")
    plt.ylabel("Value")
    plt.show()

#Hypothesis Testing (Pairwise T-tests)
for i, col1 in enumerate(numerical_columns):
    for col2 in numerical_columns[i+1:]:
        stat, p_value = ttest_ind(df_winsorized[col1], df_winsorized[col2])
        print(f"T-test between {col1} and {col2}: p-value = {p_value}")

#Clustering
numeric_columns = df_winsorized.select_dtypes(include=['float64', 'int64']).columns

# Standardize the numeric data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_winsorized[numeric_columns])

# Perform clustering for a range of cluster numbers
for n_clusters in range(2, 9):  # Example: Trying 2 to 4 clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    # Fit KMeans on scaled numeric data
    kmeans.fit(df_scaled)
    
    # Add cluster labels to the original DataFrame
    df_winsorized[f'Cluster_{n_clusters}'] = kmeans.labels_
    
    # Visualize clusters using PCA for dimensionality reduction
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df_scaled)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=kmeans.labels_, palette='viridis', s=50)
    plt.title(f"K-Means Clustering with {n_clusters} Clusters")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Cluster")
    plt.show()

# Print the cluster labels for inspection
for n_clusters in range(2, 9):
    print(f"\nCluster Labels for {n_clusters} Clusters:")
    print(df_winsorized[f'Cluster_{n_clusters}'].value_counts())
    
    
    
# Helper function for plotting cumulative percentage (Pareto chart)
def pareto_chart(data, column):
    sorted_data = data[column].sort_values(ascending=False)
    cumulative = sorted_data.cumsum() / sorted_data.sum() * 100
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_data.index, sorted_data, color='skyblue', label='Value')
    plt.plot(sorted_data.index, cumulative, color='red', label='Cumulative Percentage')
    plt.axhline(80, color='green', linestyle='--', label='80% Line')
    plt.title(f"Pareto Chart for {column}")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
    
#Pareto Chart (For each numerical column)
for column in numerical_columns:
    pareto_chart(df_winsorized, column)


#Correlation Coffecient (Business Insights)
# Ensure date columns are converted to datetime type
if "date" in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Remove non-numeric columns (like the date column) for correlation calculation
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Compute the correlation matrix
correlation_matrix = df[numerical_columns].corr()

# Display the correlation matrix
print("\nCorrelation Coefficients between columns:")
print(correlation_matrix)

# Optionally, visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()



# Clustered bar plot for Failure_status(Categorical Column)
sns.countplot(x='Failure_status', data=df, palette='viridis')
plt.title("Failure Status Distribution")
plt.xlabel("Failure Status")
plt.ylabel("Count")
plt.show()


# Histogram for Failure_status column (Categorical Column)
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='Failure_status', kde=False, bins=3, palette='coolwarm')
plt.title("Histogram of Failure Status")
plt.xlabel("Failure Status")
plt.ylabel("Frequency")
plt.show()


# Count plot for Failure_status
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Failure_status', palette='coolwarm')
plt.title("Count of Failure Status")
plt.xlabel("Failure Status")
plt.ylabel("Count")
plt.show()


#Making This data clean to load in POwer BI

# Export df_winsorized to an Excel file
df_winsorized.to_excel(r"D:\360\Mini Project\Dataset\df_winsorized.xlsx", index=False)
print("df_winsorized exported to Excel successfully!")

# Check for missing values in the df_winsorized DataFrame
print("\nMissing values in df_winsorized:")
print(df_winsorized.isnull().sum())


# Check for outliers using boxplots for each numerical column
for column in df_winsorized.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df_winsorized[column], color='lightblue')
    plt.title(f"Boxplot for {column} (Outliers Check)")
    plt.xlabel(column)
    plt.show()
