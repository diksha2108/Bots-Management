# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 17:16:24 2025

@author: diksha
"""

import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
#print(matplotlib.__version__)
# 1) Load the dataset
# Assume you downloaded the “train_1.csv” (or equivalent) from Kaggle competition data
# For example: "./data/train_1.csv"
df_raw = pd.read_csv(r"E:\kaggle dataset\web-traffic-time-series-forecasting\train_1\train_1.csv", low_memory=False)
print("Raw shape:", df_raw.shape)
# The first column is “Page”
pages = df_raw['Page']
df_ts = df_raw.drop(columns=['Page'])
print("Time-series part shape:", df_ts.shape)

# 2) Feature engineering  
# For each page (row) we can derive features such as:
#   - mean views
#   - std dev views
#   - max views
#   - number of zero days
#   - skewness
#   - maybe last-week jump
features = pd.DataFrame()
features['page'] = pages
features['mean_views'] = df_ts.mean(axis=1)
features['std_views'] = df_ts.std(axis=1)
features['max_views'] = df_ts.max(axis=1)
features['zero_days'] = (df_ts == 0).sum(axis=1)
features['skew_views'] = df_ts.skew(axis=1)

# Compute e.g., views in last 7 days vs previous 30 days
last7 = df_ts.iloc[:, -7:].mean(axis=1)
prev30 = df_ts.iloc[:, -37:-7].mean(axis=1)
features['ratio_last7_prev30'] = last7 / (prev30 + 1e-6)

print(features.head())

# 3) Simulate label: bot vs human  
# We simulate that pages which have extremely high max_views / low diversity => “bot” traffic
def simulate_label(row):
    if (row['max_views'] > row['mean_views'] * 5) and (row['zero_days'] < row['std_views']):
        return 1  # “bot”
    else:
        return 0  # “human”

features['is_bot'] = features.apply(simulate_label, axis=1)
print("Bot/human distribution:\n", features['is_bot'].value_counts())

# 4) Prepare train/test  
X = features.drop(columns=['page','is_bot'])
y = features['is_bot']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

# 5) Train model  
rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

# 6) Evaluate  
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:,1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 7) Feature importance  
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8,6))
sns.barplot(x=feat_imp.values[:10], y=feat_imp.index[:10])
plt.title("Top 10 Feature Importances")
plt.show()

# 8) Example prediction  
example = X_test.sample(n=1, random_state=0)
print("Example features:\n", example)
print("Predicted bot probability:", rf.predict_proba(example)[:,1][0])
print("Predicted class:", rf.predict(example)[0])

