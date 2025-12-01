🏥 ICU Early Warning System — Sepsis Prediction (Machine Learning + Streamlit App)

This project builds an Early Warning System (EWS) that predicts sepsis risk in ICU patients using real clinical time-series features.
It includes:

✅ End-to-end Machine Learning pipeline
✅ EDA + Data Cleaning
✅ Model Training & Comparison
✅ Random Forest, XGBoost, LightGBM, Logistic Regression
✅ Evaluation (ROC-AUC, PR-AUC, confusion matrix)
✅ Streamlit App for real-time risk prediction
🎯 Domain: Healthcare / Clinical Data Science / ICU patient monitoring

This project is designed to demonstrate clinical ML skills, feature engineering, and deployment readiness for data science roles in Digital Health, Clinical AI & Healthcare Analytics.

✨ 1. Project Overview

Sepsis is one of the leading causes of mortality in ICUs.
Early prediction can save lives — even a 1-hour delay in treatment increases mortality significantly.

This project uses real patient data to predict whether a patient is likely to develop sepsis in the next few hours.

The system includes:

Data cleaning & imputation

Exploratory Data Analysis

Feature engineering

ML model training & comparison

Deployment using Streamlit

Saved model artifacts for reproduction

📊 2. Dataset

Dataset used: PhysioNet / Sepsis Prediction Dataset (Kaggle version)
A large ICU dataset containing:

Heart Rate (HR)

O2 Saturation

Systolic/Diastolic BP

Mean Arterial Pressure (MAP)

Respiratory Rate

ICU Length of Stay (ICULOS)

Demographics

SepsisLabel (0/1)

The dataset was cleaned and prepared in:

📁 notebooks/01_eda.ipynb

🧪 3. Exploratory Data Analysis (EDA)

Performed in 01_eda.ipynb:

✔ Missing data handling
✔ Vital signs distribution plots
✔ Class imbalance analysis (only ~1–2% positive cases)
✔ Correlation heatmap
✔ Time-based feature understanding (ICULOS trends)

These insights helped shape the model & feature engineering.

🤖 4. Machine Learning Models

Model training done in:

📁 notebooks/02_model_training.ipynb

The following models were trained & compared:

Model	ROC-AUC	PR-AUC
Random Forest	0.98	0.735
XGBoost	0.83	0.168
LightGBM	0.80	0.099
Logistic Regression	0.71	0.065
🏆 Best Model: Random Forest Classifier

Why?

Handles imbalance with class_weight

Captures nonlinear feature interactions

Excellent ROC/PR performance

Stable & interpretable

This is the model used in deployment.

🧠 5. Explainability (SHAP)

Due to the dataset size & kernel limits, SHAP was computed on a small sample.

The top clinically meaningful features influencing sepsis prediction were:

Heart Rate (HR)

Respiratory Rate

Oxygen Saturation

Mean Arterial Pressure (MAP)

ICULOS (hours in ICU)

These align with clinical deterioration patterns.

🚀 6. Streamlit Prediction App

Deployed app file:

📁 app/icu_ews_app.py

⭐ Features:

Input ICU vitals

Model predicts risk probability of sepsis

Shows interpretation text

Uses saved model + feature means

Runs instantly on any system via Streamlit

▶️ How To Run the App
pip install -r requirements.txt
streamlit run app/icu_ews_app.py

📦 7. Model Artifacts

Stored in:

📁 src/models/

Includes:

random_forest_sepsis.pkl (trained model)

feature_cols.pkl (feature order)

feature_means.json (used for clean inputs)

These files are used by the Streamlit app for fast inference.

📁 8. Project Structure
ICU-Early-Warning-System/
│
├── app/
│   └── icu_ews_app.py
│
├── data/
│   └── Dataset.csv  (optional / ignored)
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_model_training.ipynb
│
├── src/
│   └── models/
│       ├── random_forest_sepsis.pkl
│       ├── feature_cols.pkl
│       └── feature_means.json
│
├── README.md
├── requirements.txt
└── .gitignore

🧰 9. Tech Stack

Python

Pandas, NumPy

Scikit-Learn

XGBoost, LightGBM

Matplotlib, Seaborn

SHAP

Streamlit (deployment)

🎯 10. Why This Project Is Recruiter-Friendly

This project demonstrates:

✔ Experience with real ICU clinical data
✔ Ability to build a complete ML pipeline
✔ Knowledge of imbalanced data handling
✔ Skill with model evaluation beyond accuracy
✔ Deployment using Streamlit
✔ Domain understanding in sepsis & ICU care
✔ Clear documentation & reproducibility

Exactly what Clinical Data Science / Digital Health / Healthcare Analytics hiring teams look for.

📝 11. Future Improvements

Add LSTM/GRU time-series model

Deploy app via Docker

Build SHAP for full dataset (GPU)

Add early-warning alert threshold curves