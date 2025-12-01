
# 🏥 ICU Early Warning System — Sepsis Prediction (Machine Learning + Streamlit)

This project builds an **Early Warning System (EWS)** that predicts **sepsis risk** in ICU patients using real clinical time-series features.

It includes:

- ✅ End-to-end Machine Learning pipeline  
- ✅ EDA + Data Cleaning  
- ✅ Model Training & Comparison  
- ✅ Random Forest, XGBoost, LightGBM, Logistic Regression  
- ✅ Evaluation (ROC-AUC, PR-AUC, Confusion Matrix)  
- ✅ Streamlit App for real-time risk prediction  

🎯 **Domain:** Healthcare · Clinical Data Science · ICU Patient Monitoring  

This project demonstrates strong skills in clinical ML, feature engineering, and ML deployment for real-time decision support.

---

# ✨ 1. Project Overview

Sepsis is one of the **leading causes of ICU mortality**.  
Even a **1-hour delay** in treatment significantly increases death risk.

This system predicts whether a patient is likely to develop **sepsis within the next few hours**, enabling early intervention.

The system includes:

- Data cleaning & imputation  
- Exploratory Data Analysis (EDA)  
- Feature engineering  
- Machine learning model development  
- A Streamlit-based deployment-ready prediction app  
- Saved model artifacts for reproduction  

---

# 📊 2. Dataset

Dataset used: **PhysioNet / Sepsis Prediction (Kaggle version)**  
Contains ICU measurements such as:

- Heart Rate (HR)  
- O2 Saturation  
- Blood Pressure (SBP, DBP, MAP)  
- Respiratory Rate  
- ICU Length of Stay (ICULOS)  
- Demographics  
- **SepsisLabel (0/1)**

Dataset cleaning & preparation done in:

📁 `notebooks/01_eda.ipynb`

---

# 🧪 3. Exploratory Data Analysis (EDA)

Performed in `01_eda.ipynb`  
Includes:

- ✔ Missing data handling  
- ✔ Vital signs distributions  
- ✔ Class imbalance (only ~1–2% positive cases)  
- ✔ Correlation heatmap  
- ✔ Time-series feature patterns (e.g., ICULOS trends)  

These insights guided feature engineering and model selection.

---

# 🤖 4. Machine Learning Models

Training done in:

📁 `notebooks/02_model_training.ipynb`

| Model | ROC-AUC | PR-AUC |
|-------|---------|--------|
| **Random Forest** | **0.98** | **0.735** |
| XGBoost | 0.83 | 0.168 |
| LightGBM | 0.80 | 0.099 |
| Logistic Regression | 0.71 | 0.065 |

🏆 **Best Model: Random Forest Classifier**

Why Random Forest?

- Handles class imbalance via `class_weight`
- Captures nonlinear interactions
- Excellent ROC/PR performance
- Stable & interpretable

This is the model used in the Streamlit deployment.

---

# 🧠 5. Explainability (SHAP)

SHAP was used on a sampled subset (due to dataset size).

**Top clinically relevant features:**

- Heart Rate (HR)  
- Respiratory Rate  
- Oxygen Saturation  
- Mean Arterial Pressure (MAP)  
- ICULOS (time spent in ICU)  

These align with real clinical deterioration patterns.

---

# 🚀 6. Streamlit Prediction App

App file:

📁 `app/icu_ews_app.py`

⭐ Features:

- Input real ICU vitals  
- Model predicts sepsis probability  
- Provides interpretation text  
- Uses saved model + feature means  
- Fast & lightweight inference  

### ▶️ Run the App

```bash
pip install -r requirements.txt
streamlit run app/icu_ews_app.py
📦 7. Model Artifacts

Stored in:

📁 src/models/

Includes:

random_forest_sepsis.pkl — trained model

feature_cols.pkl — feature ordering

feature_means.json — mean values for preprocessing

Used by the Streamlit app for real-time predictions.

📁 8. Project Structure
ICU-Early-Warning-System/
│
├── app/
│   └── icu_ews_app.py
│
├── data/
│   └── Dataset.csv  (ignored in Git)
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

Seaborn, Matplotlib

SHAP

Streamlit

🎯 10. Why This Project Is Recruiter-Friendly

This project demonstrates:

✔ Experience with real ICU clinical datasets
✔ Strong ML engineering skills
✔ Handling imbalanced clinical data
✔ Model evaluation beyond accuracy
✔ Deployment readiness via Streamlit
✔ SHAP-based explainability
✔ Clear documentation
✔ Healthcare domain knowledge

This aligns perfectly with roles in:
Clinical AI · Digital Health · Healthcare Analytics · ML Engineering

📝 11. Future Improvements

Add L
STM/GRU time-series deep learning model

Deploy via Docker or cloud (AWS/GCP/Streamlit Cloud)

Full SHAP on GPU-enabled environment

Early-warning alerts with threshold tuning

⭐ If you find this project helpful, please consider starring ⭐ the repository!




