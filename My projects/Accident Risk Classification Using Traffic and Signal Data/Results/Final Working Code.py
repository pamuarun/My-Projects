# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 19:58:49 2025
@author: Arun Teja
"""

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from imblearn.ensemble import BalancedBaggingClassifier, RUSBoostClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
from sklearn.tree import DecisionTreeClassifier
import joblib
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")  # Ignores all warnings

df1 = pd.read_csv(r"D:\360\Project  - 3\Datasets\Main Dataset\Project_dataset.csv")
df1.dtypes
# Load Dataset
df = pd.read_csv(r"D:\360\Project  - 3\Datasets\Main Dataset\Project_dataset_updated.csv")
df.dtypes

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


#EDA

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

# (Optional) Save results
#num_stats.to_csv("numerical_statistics.csv")
#cat_stats.to_csv("categorical_statistics.csv")


# Set plot style
sns.set(style="whitegrid")

#Plots

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


#Data Preprocessing

# Check for duplicates and drop them
print("\nChecking for duplicates:")
duplicates = df.duplicated().sum()
print(f"Total duplicate rows: {duplicates}")

# Drop duplicates if any
df.drop_duplicates(inplace=True)


'''
Many numeric features are skewed

vehicle_count_per_hr → skew = 3.99

avg_speed_kmph → skew = 1.67

violations_count → skew = 1.18

accident_occurred → skew = 4.33
'''

# Check missing values before
print("Missing values before imputation:\n")
print(df.isnull().sum())

# Separate numerical & categorical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(exclude=['int64', 'float64']).columns  # all non-numerics

# Numerical Imputation (single imputer: median better for skewed) 
num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Categorical Imputation (mode per column)
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[[col]] = cat_imputer.fit_transform(df[[col]].astype(str))  # force string

# Verify 
print("\nMissing values after imputation:\n")
print(df.isnull().sum())


# Winsorization

# Step 1: Identify numeric columns
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Step 2: Drop low-variation columns (like binary flags and time-related columns)
low_var_cols = ['accident_occurred']  # all low-variation columns
winsor_cols = [col for col in numerical_columns if col not in low_var_cols]

# Step 3: Apply Winsorization only on valid columns
winsor = Winsorizer(
    capping_method='iqr',  
    tail='both',
    fold=1.5,
    variables=winsor_cols
)

df_winsorized = winsor.fit_transform(df[winsor_cols])

# Step 4: Add back all excluded low-variation columns
df_final = pd.concat([df_winsorized, df[low_var_cols]], axis=1)

# Step 5: Compare boxplots before & after Winsorization
for column in winsor_cols:
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.boxplot(x=df[column], color='skyblue')
    plt.title(f'Original Boxplot - {column}')

    plt.subplot(1, 2, 2)
    sns.boxplot(x=df_final[column], color='orange')
    plt.title(f'Winsorized Boxplot - {column}')

    plt.tight_layout()
    plt.show()

# Step 6: Summary statistics
print("\nSummary Statistics After Winsorization:")
print(df_final.describe())

# Step 7: Save final cleaned dataset
#df_final.to_csv('cleaned_data_final.csv', index=False)

df_final = pd.concat([df_winsorized, df[low_var_cols]], axis=1)

# One-hot encoding
ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
encoded = pd.DataFrame(ohe.fit_transform(df[cat_cols]), 
                       columns=ohe.get_feature_names_out(cat_cols),
                       index=df.index)
df_final = df_final.join(encoded)

# Models & Hyperparameters
models = {
    "Logistic Regression": {"model": LogisticRegression(max_iter=5000, random_state=42, class_weight='balanced'), "params": {"C": [0.1, 1, 10], "solver": ["lbfgs", "liblinear"]}},
    "Balanced Bagging": {"model": BalancedBaggingClassifier(random_state=42, n_estimators=50), "params": {"max_samples": [0.5, 0.75, 1.0], "max_features": [0.5, 0.75, 1.0]}},
    "Random Forest": {"model": RandomForestClassifier(random_state=42, class_weight='balanced'), "params": {"n_estimators":[100,200], "max_depth":[None,5,10]}},
    "Gradient Boosting": {"model": GradientBoostingClassifier(random_state=42), "params": {"n_estimators":[100,200], "learning_rate":[0.01,0.1], "max_depth":[3,5]}},
    "XGBoost": {"model": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), "params": {"n_estimators": [100, 200], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1], "scale_pos_weight": [len(df_final[df_final['accident_occurred']==0])/len(df_final[df_final['accident_occurred']==1])] }},
    "LightGBM": {"model": LGBMClassifier(random_state=42, class_weight='balanced'), "params": {"n_estimators":[100,200], "learning_rate":[0.01,0.1], "num_leaves":[31,50]}},
    "CatBoost": {"model": CatBoostClassifier(verbose=0, random_state=42), "params": {"iterations":[100,200], "learning_rate":[0.01,0.1], "depth":[3,5]}},
    "RUSBoost": {"model": RUSBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42), n_estimators=100, learning_rate=0.01, random_state=42), "params": {}},
    "KNN": {"model": KNeighborsClassifier(), "params": {"n_neighbors": [5, 7, 9], "weights": ["uniform", "distance"]}},
    "Naive Bayes": {"model": GaussianNB(), "params": {}},
    "MLP": {"model": MLPClassifier(max_iter=500, random_state=42), "params": {"hidden_layer_sizes": [(50,), (50,50)], "activation": ["relu","tanh"], "alpha":[0.0001,0.001]}}
}

# Scalers
scalers = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler(), "RobustScaler": RobustScaler(), "MaxAbsScaler": MaxAbsScaler()}

# Resampling Methods
resamplers = {
    "RandomOverSampler": RandomOverSampler(sampling_strategy=1.0, random_state=42),
    "SMOTE": SMOTE(sampling_strategy=1.0, random_state=42, k_neighbors=5),
    "BorderlineSMOTE": BorderlineSMOTE(sampling_strategy=1.0, random_state=42, kind="borderline-1"),
    "ADASYN": ADASYN(sampling_strategy=1.0, random_state=42, n_neighbors=5)
}

# Main Loop with Status Prints
results = []
colors = ['b','g','r','c','m','y','k','orange','purple','brown','gray','pink']
plt.figure(figsize=(14,6))
color_idx = 0

X = df_final.drop("accident_occurred", axis=1)
y = df_final["accident_occurred"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

for scaler_name, scaler in scalers.items():
    df_scaled = df_final.copy()
    continuous_cols = [col for col in df_scaled.select_dtypes(include=['float64','int64']).columns 
                       if col != "accident_occurred" and df_scaled[col].nunique() > 10]
    df_scaled[continuous_cols] = scaler.fit_transform(df_scaled[continuous_cols])
    
    X_scaled = df_scaled.drop("accident_occurred", axis=1)
    y_scaled = df_scaled["accident_occurred"]
    
    for res_name, resampler in resamplers.items():
        X_train_res, y_train_res = resampler.fit_resample(X_train, y_train)
        
        for model_name, mp in models.items():
            # Print current status
            print(f"\n🔹 Training Model: {model_name} | Scaler: {scaler_name} | Resampler: {res_name}")
            
            clf = GridSearchCV(mp["model"], mp["params"], cv=StratifiedKFold(n_splits=5), scoring='f1', n_jobs=-1) if mp["params"] else mp["model"]
            clf.fit(X_train_res, y_train_res)
            best_model = clf.best_estimator_ if hasattr(clf, "best_estimator_") else clf
            best_params = clf.best_params_ if hasattr(clf, "best_params_") else "Default"
            
            y_prob = best_model.predict_proba(X_test)[:,1] if hasattr(best_model, "predict_proba") else best_model.decision_function(X_test)
            if not hasattr(best_model, "predict_proba"):
                y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
            y_pred_final = (y_prob > 0.5).astype(int)
            
            train_acc = best_model.score(X_train_res, y_train_res)
            test_acc = (y_pred_final == y_test).mean()
            prec = precision_score(y_test, y_pred_final, zero_division=0)
            rec = recall_score(y_test, y_pred_final, zero_division=0)
            f1_val = f1_score(y_test, y_pred_final, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_prob)
            precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
            pr_auc = auc(recall_vals, precision_vals)
            
            results.append({
                "Model": model_name,
                "Scaler": scaler_name,
                "Resampler": res_name,
                "Best_Params": best_params,
                "Threshold": 0.5,
                "Training_Accuracy": train_acc,
                "Testing_Accuracy": test_acc,
                "Precision": prec,
                "Recall": rec,
                "F1": f1_val,
                "ROC-AUC": roc_auc,
                "PR-AUC": pr_auc
            })
            
            RocCurveDisplay.from_predictions(y_test, y_prob, name=f"{model_name}-{scaler_name}-{res_name}", color=colors[color_idx % len(colors)], ax=plt.gca())
            color_idx += 1

plt.title("ROC Curves for All Models + Scalers + Resamplers")
plt.show()

# Results Summary
results_df = pd.DataFrame(results)
best_model_info = results_df.sort_values(by="F1", ascending=False).iloc[0]

# Save results summary
results_df.to_csv("all_models_results.csv", index=False)

# Save best model info
best_model_info.to_frame().T.to_csv("best_model_summary.csv", index=False)


print("\n===== Summary of All Models + Resamplers =====")
print(results_df)

print("\n===== Best Model Based on F1 =====")
print(best_model_info)

# Refit and Save Best Model
best_scaler_name = best_model_info['Scaler']
best_res_name = best_model_info['Resampler']
best_model_name = best_model_info['Model']

best_scaler = scalers[best_scaler_name]
best_resampler = resamplers[best_res_name]

X_train_res, y_train_res = best_resampler.fit_resample(X_train, y_train)
model_info = models[best_model_name]
best_model = model_info["model"].set_params(**best_model_info["Best_Params"]) if best_model_info["Best_Params"] != "Default" else model_info["model"]
best_model.fit(X_train_res, y_train_res)

# Save for Streamlit
joblib.dump(best_model, "best_model.pkl")
joblib.dump(best_scaler, "best_scaler.pkl")
joblib.dump(best_resampler, "best_resampler.pkl")
print("\n✅ Best model, scaler, and resampler saved as .pkl files.")

# Test Predictions
y_prob_test = best_model.predict_proba(X_test)[:,1] if hasattr(best_model, "predict_proba") else best_model.decision_function(X_test)
if not hasattr(best_model, "predict_proba"):
    y_prob_test = (y_prob_test - y_prob_test.min()) / (y_prob_test.max() - y_prob_test.min())

y_pred_best = (y_prob_test > 0.5).astype(int)

comparison_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred_best,
    "Probabilities": y_prob_test
})

comparison_df.to_csv("Actual_vs_Predicted_FixedThreshold.csv", index=False)

print("\n===== Sample Predictions =====")
print(comparison_df.head(20))

comparison_df['Correct'] = comparison_df['Actual'] == comparison_df['Predicted']
print("\n===== Prediction Accuracy Summary =====")
print(comparison_df['Correct'].value_counts())

# Sample predictions on test set
sample = X_test.sample(350, random_state=42)

sample_pred_prob = best_model.predict_proba(sample)[:,1] if hasattr(best_model, "predict_proba") else best_model.decision_function(sample)
if not hasattr(best_model, "predict_proba"):
    sample_pred_prob = (sample_pred_prob - sample_pred_prob.min()) / (sample_pred_prob.max() - sample_pred_prob.min())

threshold = 0.5
sample_pred_class = (sample_pred_prob > threshold).astype(int)

sample_results = sample.copy()
sample_results['Predicted_Prob'] = sample_pred_prob
sample_results['Predicted_Class'] = sample_pred_class

print("\n===== Sample Testing Predictions =====")
print(sample_results)


