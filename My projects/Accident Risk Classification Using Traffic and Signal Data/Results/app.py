# -*- coding: utf-8 -*-
"""
Accident Prediction Dashboard (Enhanced & Fixed)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score, PrecisionRecallDisplay, RocCurveDisplay

# ======================
# Page Config
# ======================
st.set_page_config(page_title="Accident Prediction Dashboard", layout="wide")
st.title("🚦 Accident Prediction Dashboard")

# ======================
# Load Data
# ======================
comparison_df = pd.read_csv(r"D:\360\Project  - 3\Results\Actual_vs_Predicted_FixedThreshold.csv")
comparison_df['Correct'] = comparison_df['Actual'] == comparison_df['Predicted']

# ======================
# Sidebar Filters
# ======================
st.sidebar.header("🔎 Filters")

prediction_filter = st.sidebar.multiselect(
    "Prediction Type",
    options=["Correct", "Incorrect"],
    default=["Correct", "Incorrect"]
)

min_prob, max_prob = st.sidebar.slider(
    "Prediction Probability Range",
    0.0, 1.0, (0.0, 1.0), 0.01
)

# Weather filter
if 'weather' in comparison_df.columns:
    weather_filter = st.sidebar.multiselect(
        "Weather Condition",
        options=comparison_df['weather'].unique(),
        default=list(comparison_df['weather'].unique())
    )
else:
    weather_filter = None  # fallback

# Apply filters
df_filtered = comparison_df[
    ((comparison_df['Correct'] & ("Correct" in prediction_filter)) |
     (~comparison_df['Correct'] & ("Incorrect" in prediction_filter))) &
    (comparison_df['Probabilities'] >= min_prob) &
    (comparison_df['Probabilities'] <= max_prob)
].copy()

if weather_filter is not None:
    df_filtered = df_filtered[df_filtered['weather'].isin(weather_filter)]

# ======================
# Risk Level
# ======================
def risk_level(prob):
    if prob >= 0.75:
        return "High Risk 🔴"
    elif prob >= 0.5:
        return "Medium Risk 🟡"
    else:
        return "Low Risk 🟢"

df_filtered['Risk_Level'] = df_filtered['Probabilities'].apply(risk_level)

# ======================
# Metrics Summary
# ======================
st.subheader("✅ Prediction Metrics Summary")
accuracy = df_filtered['Correct'].mean() if len(df_filtered) > 0 else 0
recall = recall_score(df_filtered['Actual'], df_filtered['Predicted'], zero_division=0) if len(df_filtered) > 0 else 0
roc_auc = roc_auc_score(df_filtered['Actual'], df_filtered['Probabilities']) if len(df_filtered) > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{accuracy:.2%}")
col2.metric("Recall", f"{recall:.2%}")
col3.metric("ROC-AUC", f"{roc_auc:.2%}")

# ======================
# Domain & Feature Insights
# ======================
st.subheader("🌤 Domain & Feature Insights")

if 'weather' in df_filtered.columns:
    st.write("**Weather Distribution:**")
    st.bar_chart(df_filtered['weather'].value_counts())

if 'avg_speed_kmph' in df_filtered.columns:
    st.metric("Average Speed (kmph)", f"{df_filtered['avg_speed_kmph'].mean():.2f}")

if 'violations_count' in df_filtered.columns:
    st.metric("Average Violations Count", f"{df_filtered['violations_count'].mean():.2f}")

if 'road_type' in df_filtered.columns:
    st.write("**Road Type Distribution:**")
    st.bar_chart(df_filtered['road_type'].value_counts())

st.write("**Risk Level Distribution:**")
st.bar_chart(df_filtered['Risk_Level'].value_counts())

# ======================
# Conditional Styling Functions
# ======================
def highlight_row(row):
    color = '#0D47A1' if row['Correct'] == '✅' else '#FF5252'
    return [f'background-color: {color}; color: white' for _ in row]

def highlight_prob(val):
    if val >= 0.75:
        return 'background-color: #0D47A1; color: white'
    elif val >= 0.5:
        return 'background-color: #FFEB3B; color: black'
    else:
        return 'background-color: #FF5252; color: white'

def highlight_tick(val):
    return 'background-color: black; color: white; font-weight:bold; text-align:center' if val in ['✅','❌'] else ''

# Replace Correct column with tick symbols
df_filtered['Correct'] = df_filtered['Correct'].apply(lambda x: '✅' if x else '❌')

# Apply styling
styled_df = df_filtered.style.apply(highlight_row, axis=1)\
                             .applymap(highlight_prob, subset=['Probabilities'])\
                             .applymap(highlight_tick, subset=['Correct'])

# ======================
# Display Predictions Table
# ======================
st.subheader("📊 Predictions Table with Risk & Probabilities")
st.dataframe(styled_df, use_container_width=True)

# ======================
# Download Filtered Data
# ======================
csv = df_filtered.to_csv(index=False)
st.download_button("📥 Download Filtered Data with Risk Levels", csv, "filtered_predictions.csv", "text/csv")

# ======================
# Top N Predictions
# ======================
st.subheader("⚠ Top 5 Highest Probability Predictions")
st.dataframe(df_filtered.nlargest(5, 'Probabilities'))

st.subheader("✅ Top 5 Lowest Probability Predictions")
st.dataframe(df_filtered.nsmallest(5, 'Probabilities'))

# ======================
# Probability Distribution
# ======================
st.subheader("📈 Distribution of Predicted Probabilities")
fig, ax = plt.subplots(figsize=(10,4))
sns.histplot(df_filtered['Probabilities'], bins=30, kde=True, color='skyblue', ax=ax)
ax.set_title("Prediction Probabilities Distribution")
st.pyplot(fig)

# ======================
# Confusion Matrix Heatmap
# ======================
st.subheader("🧾 Confusion Matrix Heatmap")
cm = confusion_matrix(df_filtered['Actual'], df_filtered['Predicted'])
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# ======================
# ROC Curve & Precision-Recall Curve
# ======================
st.subheader("📉 ROC Curve")
fig, ax = plt.subplots()
RocCurveDisplay.from_predictions(df_filtered['Actual'], df_filtered['Probabilities'], ax=ax)
st.pyplot(fig)

st.subheader("📊 Precision-Recall Curve")
fig, ax = plt.subplots()
PrecisionRecallDisplay.from_predictions(df_filtered['Actual'], df_filtered['Probabilities'], ax=ax)
st.pyplot(fig)
