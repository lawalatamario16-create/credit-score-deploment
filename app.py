"""
app.py
======
Streamlit app untuk deployment model Credit_Score (bagian 1c).

Menu:
  1. Prediksi Manual  -> isi form data 1 nasabah, lihat prediksi + probabilitas
  2. Prediksi Batch    -> upload CSV berisi banyak nasabah, download hasil

Jalankan dengan:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd

from predict import CreditScorePredictor

st.set_page_config(page_title="Credit Score Predictor", page_icon="💳", layout="centered")

ARTIFACTS_DIR = "."


@st.cache_resource
def load_predictor():
    return CreditScorePredictor(artifacts_dir=ARTIFACTS_DIR)


predictor = load_predictor()

OCCUPATIONS = ['Accountant', 'Architect', 'Developer', 'Doctor', 'Engineer', 'Entrepreneur',
               'Journalist', 'Lawyer', 'Manager', 'Mechanic', 'Media_Manager', 'Musician',
               'Scientist', 'Teacher', 'Writer']
CREDIT_MIX = ['Bad', 'Standard', 'Good']
PAY_MIN_AMOUNT = ['Yes', 'No', 'NM']
PAY_BEHAVIOUR = ['Low_spent_Small_value_payments', 'Low_spent_Medium_value_payments',
                  'Low_spent_Large_value_payments', 'High_spent_Small_value_payments',
                  'High_spent_Medium_value_payments', 'High_spent_Large_value_payments']

LABEL_COLOR = {"Good": "🟢", "Standard": "🟡", "Poor": "🔴"}

st.title("💳 Credit Score Predictor")
st.caption("Deployment model klasifikasi Credit_Score (Good / Standard / Poor) — bagian 1c")

st.subheader("Masukkan Data Nasabah")

with st.form("manual_form"):
    c1, c2 = st.columns(2)

    with c1:
        age = st.number_input("Age", min_value=14, max_value=100, value=30)
        annual_income = st.number_input("Annual Income", min_value=0.0, value=45000.0, step=1000.0)
        monthly_salary = st.number_input("Monthly Inhand Salary", min_value=0.0, value=3500.0, step=100.0)
        num_bank_accounts = st.number_input("Num Bank Accounts", min_value=0, max_value=20, value=3)
        num_credit_card = st.number_input("Num Credit Card", min_value=0, value=4)
        interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=40.0, value=12.0)
        num_loan = st.number_input("Num of Loan", min_value=0, max_value=20, value=2)
        delay_due = st.number_input("Delay from Due Date (hari)", min_value=0, value=5)
        num_delayed = st.number_input("Num of Delayed Payment", min_value=0, max_value=40, value=3)
        changed_limit = st.number_input("Changed Credit Limit", value=2.5, step=0.1)

    with c2:
        num_inquiries = st.number_input("Num Credit Inquiries", min_value=0, value=4)
        outstanding_debt = st.number_input("Outstanding Debt", min_value=0.0, value=1200.0, step=100.0)
        credit_util = st.number_input("Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0, value=32.5)
        emi = st.number_input("Total EMI per Month", min_value=0.0, value=150.0, step=10.0)
        invested = st.number_input("Amount Invested Monthly", min_value=0.0, value=200.0, step=10.0)
        monthly_balance = st.number_input("Monthly Balance", min_value=0.0, value=400.0, step=10.0)
        credit_history_months = st.number_input("Credit History Age (bulan)", min_value=0, value=120)
        num_loan_types = st.number_input("Num Loan Types", min_value=0, value=2)
        occupation = st.selectbox("Occupation", OCCUPATIONS, index=OCCUPATIONS.index("Engineer"))
        credit_mix = st.selectbox("Credit Mix", CREDIT_MIX, index=CREDIT_MIX.index("Good"))

    c3, c4 = st.columns(2)
    with c3:
        pay_min_amount = st.selectbox("Payment of Min Amount", PAY_MIN_AMOUNT, index=1)
    with c4:
        pay_behaviour = st.selectbox("Payment Behaviour", PAY_BEHAVIOUR, index=4)

    submitted = st.form_submit_button("🔮 Prediksi Credit Score", use_container_width=True)

if submitted:
    input_data = {
        "Age": age, "Annual_Income": annual_income, "Monthly_Inhand_Salary": monthly_salary,
        "Num_Bank_Accounts": num_bank_accounts, "Num_Credit_Card": num_credit_card,
        "Interest_Rate": interest_rate, "Num_of_Loan": num_loan,
        "Delay_from_due_date": delay_due, "Num_of_Delayed_Payment": num_delayed,
        "Changed_Credit_Limit": changed_limit, "Num_Credit_Inquiries": num_inquiries,
        "Outstanding_Debt": outstanding_debt, "Credit_Utilization_Ratio": credit_util,
        "Total_EMI_per_month": emi, "Amount_invested_monthly": invested,
        "Monthly_Balance": monthly_balance, "Credit_History_Age_Months": credit_history_months,
        "Num_Loan_Types": num_loan_types, "Occupation": occupation, "Credit_Mix": credit_mix,
        "Payment_of_Min_Amount": pay_min_amount, "Payment_Behaviour": pay_behaviour,
    }

    try:
        result = predictor.predict(input_data)
        label = result["prediction"]
        st.success(f"### {LABEL_COLOR.get(label, '')} Prediksi: **{label}**")

        if "probabilities" in result:
            st.write("**Probabilitas tiap kelas:**")
            proba_df = pd.DataFrame(
                result["probabilities"].items(), columns=["Kelas", "Probabilitas"]
            )
            st.bar_chart(proba_df.set_index("Kelas"))
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")

st.divider()
st.caption("Model: RandomForestClassifier · Artefak: credit_score_model.pkl, preprocessor.pkl, label_encoder.pkl, feature_schema.pkl")