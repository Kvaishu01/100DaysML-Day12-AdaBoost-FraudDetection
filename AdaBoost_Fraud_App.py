# Day 12 - AdaBoost Fraud Detection (Streamlit App)
# Run: streamlit run AdaBoost_Fraud_App.py

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score, confusion_matrix
)
import matplotlib.pyplot as plt

st.set_page_config(page_title="AdaBoost – Fraud Detection", layout="wide")

st.title("💳 AdaBoost – Fraud Detection (Day 12)")
st.write("Detect fraudulent transactions on an imbalanced dataset using **AdaBoost** with a shallow decision tree base learner.")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    n_samples = st.slider("Samples", 2000, 20000, 10000, step=1000)
    fraud_rate = st.slider("Fraud Rate (%)", 1, 10, 1)
    n_estimators = st.slider("AdaBoost n_estimators", 50, 400, 200, step=50)
    learning_rate = st.slider("AdaBoost learning_rate", 0.05, 1.0, 0.5, step=0.05)
    max_depth = st.slider("Base Tree max_depth", 1, 5, 2)
    test_size = st.slider("Test Size (%)", 10, 40, 20)
    seed = st.number_input("Random Seed", value=42, step=1)

# 1) Data
X, y = make_classification(
    n_samples=n_samples,
    n_features=20,
    n_informative=8,
    n_redundant=4,
    n_classes=2,
    weights=[1 - fraud_rate/100.0, fraud_rate/100.0],
    class_sep=1.2,
    random_state=seed,
)

feature_names = [f"feat_{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df["is_fraud"] = y

st.subheader("📂 Dataset Preview")
st.write(df.head())

# 2) Split & scale
X_train, X_test, y_train, y_test = train_test_split(
    df[feature_names], df["is_fraud"],
    test_size=test_size/100.0, stratify=df["is_fraud"], random_state=seed
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# 3) Model
base_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
model = AdaBoostClassifier(
    estimator=base_tree,
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    random_state=seed
)
model.fit(X_train_s, y_train)

# 4) Predict & metrics
y_pred = model.predict(X_test_s)
y_proba = model.predict_proba(X_test_s)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)
report = classification_report(y_test, y_pred, digits=4)

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{acc:.3f}")
col2.metric("ROC-AUC", f"{roc_auc:.3f}")
col3.metric("PR-AUC", f"{pr_auc:.3f}")

st.subheader("🧾 Classification Report")
st.text(report)

# Confusion matrix
st.subheader("🔢 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
st.write(pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"]))

# ROC curve
st.subheader("📈 ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_proba)
fig1, ax1 = plt.subplots()
ax1.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
ax1.plot([0,1],[0,1],"k--")
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.legend(loc="lower right")
st.pyplot(fig1)

# Precision-Recall curve
st.subheader("📉 Precision–Recall Curve")
prec, rec, _ = precision_recall_curve(y_test, y_proba)
fig2, ax2 = plt.subplots()
ax2.plot(rec, prec, label=f"AP = {pr_auc:.3f}")
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.legend(loc="lower left")
st.pyplot(fig2)

# Top suspicious transactions
st.subheader("🔎 Top Suspicious Transactions (by fraud probability)")
top_k = st.slider("Show top-K", 3, 20, 5)
top_idx = np.argsort(-y_proba)[:top_k]
show = X_test.iloc[top_idx].copy()
show["fraud_prob"] = y_proba[top_idx]
show["true_label"] = y_test.iloc[top_idx].values
st.dataframe(show.reset_index(drop=True))
