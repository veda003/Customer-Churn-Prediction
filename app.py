import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("models/churn_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: gray;
    margin-bottom: 30px;
}

.result-box {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📊 Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning powered customer churn prediction'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📌 About")

st.sidebar.write(
    """
    This application predicts whether a customer
    is likely to churn using a Machine Learning model.

    **Model:** Logistic Regression

    **Dataset:** Synthetic customer churn dataset

    **Features:** Customer demographics, contract,
    internet service, payment method, tenure and charges.
    """
)

st.sidebar.divider()

st.sidebar.info(
    "This prediction is a machine-learning estimate "
    "and should not be treated as a guaranteed outcome."
)


# ============================================================
# CUSTOMER INFORMATION
# ============================================================

st.subheader("👤 Customer Information")

col1, col2, col3 = st.columns(3)


# ------------------------------------------------------------
# COLUMN 1
# ------------------------------------------------------------

with col1:

    st.markdown("### Personal Details")

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )


# ------------------------------------------------------------
# COLUMN 2
# ------------------------------------------------------------

with col2:

    st.markdown("### Service Details")

    tenure = st.slider(
        "Tenure (months)",
        min_value=1,
        max_value=72,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )


# ------------------------------------------------------------
# COLUMN 3
# ------------------------------------------------------------

with col3:

    st.markdown("### Billing Details")

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Bank transfer",
            "Credit card",
            "Electronic check",
            "Mailed check"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=20.0,
        max_value=120.0,
        value=70.0,
        step=1.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=9000.0,
        value=1000.0,
        step=50.0
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Customer Churn",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    customer_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "InternetService": [internet_service],
        "Contract": [contract],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })


    # Preprocess
    processed_data = preprocessor.transform(
        customer_data
    )


    # Prediction
    prediction = model.predict(
        processed_data
    )[0]

    probability = model.predict_proba(
        processed_data
    )[0][1]


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.subheader("🎯 Prediction Result")

    result_col1, result_col2 = st.columns(2)


    with result_col1:

        if prediction == 1:

            st.error(
                "⚠️ Customer is likely to churn"
            )

        else:

            st.success(
                "✅ Customer is unlikely to churn"
            )


    with result_col2:

        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )


    # ========================================================
    # PROBABILITY BAR
    # ========================================================

    st.write("### Churn Probability")

    st.progress(
        float(probability)
    )


    # ========================================================
    # INTERPRETATION
    # ========================================================

    if probability >= 0.70:

        st.warning(
            "High predicted churn probability. "
            "This customer may require retention attention."
        )

    elif probability >= 0.50:

        st.warning(
            "Moderate predicted churn probability."
        )

    else:

        st.info(
            "Low predicted churn probability."
        )


    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    with st.expander("View Customer Information"):

        st.dataframe(
            customer_data,
            use_container_width=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Customer Churn Prediction | "
    "Python • Pandas • Scikit-learn • Streamlit"
)