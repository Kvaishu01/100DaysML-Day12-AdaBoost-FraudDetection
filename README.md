# Day 12 – AdaBoost Classifier: Fraud Detection 💳

This project detects **fraudulent transactions** using **AdaBoost** with a shallow Decision Tree as the base learner.  
Includes both a **terminal script** and a **Streamlit web app**.

## 🔍 Why AdaBoost?
AdaBoost (Adaptive Boosting) builds many **weak learners** sequentially, giving more weight to the mistakes of previous learners.  
It’s effective for **imbalanced problems** like fraud detection.

## 🧠 What’s inside?
- Synthetic, imbalanced dataset (≈1% fraud)
- Model: `AdaBoostClassifier(DecisionTree(max_depth=2))`
- Metrics: **ROC-AUC**, **PR-AUC**, **Confusion Matrix**, **Classification Report**
- Streamlit app with interactive controls + top suspicious transactions

## ▶️ Quick Start
```bash
pip install -r requirements.txt
# Terminal script
python AdaBoost_Fraud.py

# Streamlit app
streamlit run AdaBoost_Fraud_App.py
