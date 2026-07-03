"""
app.py
======
Streamlit deployment for the Credit_Score classification model
(Model Deployment, Dataset A).

Loads the artifacts produced by pipeline.py:
    - credit_score_model.pkl   (best trained model)
    - preprocessor.pkl         (fitted ColumnTransformer)
    - label_encoder.pkl        (target LabelEncoder)
    - feature_schema.pkl       (numeric_cols / categorical_cols used at fit time)

Run locally with:
    streamlit run app.py

Deploy on Streamlit Community Cloud by pointing it at this repo + this file.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Score Prediction", page_icon="💳", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load("credit_score_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    schema = joblib.load("feature_schema.pkl")
    return model, preprocessor, label_encoder, schema


model, preprocessor, label_encoder, schema = load_artifacts()
NUMERIC_COLS = schema["numeric_cols"]
CATEGORICAL_COLS = schema["categorical_cols"]

# Known categories seen during EDA (used to populate dropdowns).
CATEGORY_OPTIONS = {
    "Occupation": [
        "Developer", "Musician", "Scientist", "Entrepreneur", "Accountant",
        "Journalist", "Media_Manager", "Mechanic", "Writer", "Doctor",
        "Teacher", "Lawyer", "Engineer", "Architect",
    ],
    "Credit_Mix": ["Good", "Standard", "Bad"],
    "Payment_of_Min_Amount": ["Yes", "No", "NM"],
    "Payment_Behaviour": [
        "High_spent_Large_value_payments", "High_spent_Medium_value_payments",
        "High_spent_Small_value_payments", "Low_spent_Large_value_payments",
        "Low_spent_Medium_value_payments", "Low_spent_Small_value_payments",
    ],
}

NUMERIC_DEFAULTS = {
    "Age": 35, "Annual_Income": 50000.0, "Monthly_Inhand_Salary": 4000.0,
    "Num_Bank_Accounts": 3, "Num_Credit_Card": 4, "Interest_Rate": 12,
    "Num_of_Loan": 2, "Delay_from_due_date": 10, "Num_of_Delayed_Payment": 5,
    "Changed_Credit_Limit": 5.0, "Num_Credit_Inquiries": 3, "Outstanding_Debt": 1000.0,
    "Credit_Utilization_Ratio": 30.0, "Total_EMI_per_month": 100.0,
    "Amount_invested_monthly": 100.0, "Monthly_Balance": 300.0,
    "Credit_History_Age_Months": 120, "Num_Loan_Types": 2,
}

st.title("💳 Credit Score Prediction")
st.caption("DTSC6012001 - Model Deployment | Dataset A | Model deployment demo")

st.markdown("Isi data nasabah di bawah, lalu klik **Predict** untuk melihat hasil klasifikasi Credit Score.")

with st.form("input_form"):
    st.subheader("Data Numerik")
    num_col1, num_col2 = st.columns(2)
    numeric_inputs = {}
    for i, col in enumerate(NUMERIC_COLS):
        target_col = num_col1 if i % 2 == 0 else num_col2
        default = NUMERIC_DEFAULTS.get(col, 0.0)
        numeric_inputs[col] = target_col.number_input(
            col.replace("_", " "), value=float(default), step=1.0, format="%.2f"
        )

    st.subheader("Data Kategorikal")
    cat_col1, cat_col2 = st.columns(2)
    categorical_inputs = {}
    for i, col in enumerate(CATEGORICAL_COLS):
        target_col = cat_col1 if i % 2 == 0 else cat_col2
        options = CATEGORY_OPTIONS.get(col, [""])
        categorical_inputs[col] = target_col.selectbox(col.replace("_", " "), options)

    submitted = st.form_submit_button("🔍 Predict", use_container_width=True)


if submitted:
    row = {**numeric_inputs, **categorical_inputs}
    X_input = pd.DataFrame([row])[NUMERIC_COLS + CATEGORICAL_COLS]

    X_proc = preprocessor.transform(X_input)
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()

    pred_encoded = model.predict(X_proc)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_proc)[0]

    st.divider()
    color = {"Good": "green", "Standard": "orange", "Poor": "red"}.get(pred_label, "blue")
    st.markdown(f"### Hasil Prediksi: :{color}[**{pred_label}**]")

    if proba is not None:
        proba_df = pd.DataFrame({
            "Credit_Score": label_encoder.classes_,
            "Probability": proba,
        }).sort_values("Probability", ascending=False).reset_index(drop=True)
        st.bar_chart(proba_df.set_index("Credit_Score"))
        st.dataframe(proba_df, hide_index=True, use_container_width=True)

    with st.expander("Lihat input yang dikirim ke model"):
        st.dataframe(X_input, use_container_width=True)

st.divider()
st.caption("Model: best model dari pipeline.py (dipilih otomatis berdasarkan F1-macro tertinggi via MLflow).")
