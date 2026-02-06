# streamlit/app.py

"""
Streamlit dashboard for interactive churn predictions.

Calls the FastAPI backend for all predictions instead of loading
model artifacts directly.

"""

import streamlit as st
import pandas as pd
import requests

API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
)

st.title("SyriaTel Customer Churn Predictor")
st.markdown("Predict whether a customer is likely to churn based on their usage data.")

with st.expander("Key Churn Insights", expanded=False):
    st.caption("Based on analysis of 3,333 SyriaTel customers. Overall churn rate: 14.5%.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            "**Customer Service Calls**\n\n"
            "Customers with **4+ service calls** churn at **51.7%** — "
            "over 3x the baseline rate of 14.5%."
        )
        st.markdown(
            "**Voice Mail Plan**\n\n"
            "Voice mail subscribers churn at only **8.7%**, "
            "nearly half the rate of non-subscribers (16.7%)."
        )

    with col2:
        st.markdown(
            "**International Plan**\n\n"
            "International plan holders churn at **42.4%**, "
            "compared to just **11.5%** for those without."
        )
        st.markdown(
            "**High Day Charges**\n\n"
            "Customers in the top 25% of day charges churn at **29.3%**, "
            "over double the rate of bottom 25% spenders (12.2%)."
        )

# Check API health
try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    model_loaded = health["model_loaded"]
    if not model_loaded:
        st.error("Model not loaded on the API server. Run the training script first: `uv run train`")
except requests.ConnectionError:
    model_loaded = False
    st.error(f"Cannot connect to API at {API_URL}. Start the API first: `uv run uvicorn api.main:app --port 8000`")

# Sidebar: input form
st.sidebar.header("Customer Features")

with st.sidebar.form("prediction_form"):
    customer_name = st.text_input(
        "Customer Name (optional)",
        value="",
        help="For identification purposes only — not used by the model.",
    )

    st.subheader("High-Impact Features")

    customer_service_calls = st.number_input(
        "Customer Service Calls", min_value=0, max_value=20, value=1,
        help="Avg: 1.6. Values above 4 strongly indicate churn.",
    )
    total_day_charge = st.number_input(
        "Total Day Charge ($)", min_value=0.0, max_value=100.0, value=30.0, step=0.5,
        help="Avg: $30.56. Range: $0 – $75.90.",
    )
    international_plan = st.selectbox(
        "International Plan", ["no", "yes"],
        help="Customers with an international plan churn at a much higher rate.",
    )
    total_eve_charge = st.number_input(
        "Total Evening Charge ($)", min_value=0.0, max_value=50.0, value=17.0, step=0.5,
        help="Avg: $17.08. Range: $0 – $35.50.",
    )
    total_intl_charge = st.number_input(
        "Total International Charge ($)", min_value=0.0, max_value=10.0, value=2.70, step=0.1,
        help="Avg: $2.76. Range: $0 – $5.40.",
    )
    total_intl_calls = st.number_input(
        "Total International Calls", min_value=0, max_value=20, value=3,
        help="Avg: 4.5 calls. Range: 0 – 20.",
    )
    total_night_charge = st.number_input(
        "Total Night Charge ($)", min_value=0.0, max_value=50.0, value=11.0, step=0.5,
        help="Avg: $9.04. Range: $0 – $17.77.",
    )
    voice_mail_plan = st.selectbox(
        "Voice Mail Plan", ["no", "yes"],
        help="Customers with voicemail tend to churn less.",
    )

    st.subheader("Additional Features")

    state = st.text_input(
        "State (abbreviation)", value="OH", max_chars=2,
        help="Two-letter US state code, e.g. KS, OH, NJ.",
    )
    account_length = st.number_input(
        "Account Length (days)", min_value=0, max_value=500, value=100,
        help="Avg: 101 days. Range: 1 – 243.",
    )
    area_code = st.selectbox(
        "Area Code", [408, 415, 510], index=1,
        help="Telephone area code (408, 415, or 510).",
    )
    number_vmail_messages = st.number_input(
        "Voicemail Messages", min_value=0, max_value=60, value=0,
        help="Avg: 8.1. Non-zero only if voice mail plan is 'yes'.",
    )
    total_day_calls = st.number_input(
        "Total Day Calls", min_value=0, max_value=200, value=100,
        help="Avg: 100 calls. Range: 0 – 165.",
    )
    total_eve_calls = st.number_input(
        "Total Evening Calls", min_value=0, max_value=200, value=100,
        help="Avg: 100 calls. Range: 0 – 170.",
    )
    total_night_calls = st.number_input(
        "Total Night Calls", min_value=0, max_value=200, value=100,
        help="Avg: 100 calls. Range: 0 – 175.",
    )

    submitted = st.form_submit_button("Predict Churn", use_container_width=True)

# Main area: prediction results
if submitted and model_loaded:
    payload = {
        "state": state.upper(),
        "account_length": account_length,
        "area_code": area_code,
        "international_plan": international_plan,
        "voice_mail_plan": voice_mail_plan,
        "number_vmail_messages": number_vmail_messages,
        "total_day_calls": total_day_calls,
        "total_day_charge": total_day_charge,
        "total_eve_calls": total_eve_calls,
        "total_eve_charge": total_eve_charge,
        "total_night_calls": total_night_calls,
        "total_night_charge": total_night_charge,
        "total_intl_calls": total_intl_calls,
        "total_intl_charge": total_intl_charge,
        "customer_service_calls": customer_service_calls,
    }

    response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    result = response.json()

    # Display result
    st.divider()

    if customer_name.strip():
        st.subheader(f"Results for {customer_name.strip()}")

    col1, col2, col3 = st.columns(3)

    with col1:
        if result["churn"]:
            st.error("**Prediction: CHURN**")
        else:
            st.success("**Prediction: NO CHURN**")

    with col2:
        st.metric("Churn Probability", f"{result['churn_probability']:.1%}")

    with col3:
        risk = result["risk_level"]
        risk_colors = {"low": "🟢", "medium": "🟡", "high": "🔴"}
        st.metric("Risk Level", f"{risk_colors.get(risk, '')} {risk.upper()}")

    # Probability bar
    st.progress(result["churn_probability"], text=f"Churn probability: {result['churn_probability']:.1%}")

    # Show input summary
    with st.expander("View input data"):
        st.json(payload)

# Batch prediction via CSV upload
st.divider()
st.subheader("Batch Prediction")
st.markdown("Upload a CSV file with customer data to predict churn for multiple customers at once.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None and model_loaded:
    df = pd.read_csv(uploaded_file)
    st.write(f"Loaded **{len(df)} customers**")
    st.dataframe(df.head(), use_container_width=True)

    if st.button("Run Batch Prediction", use_container_width=True):
        records = df.to_dict(orient="records")
        customers = []
        for record in records:
            customers.append({
                "state": str(record.get("State", "OH")),
                "account_length": int(record.get("Account_Length", 100)),
                "area_code": int(record.get("Area_Code", 415)),
                "international_plan": str(record.get("International_Plan", "no")),
                "voice_mail_plan": str(record.get("Voice_Mail_Plan", "no")),
                "number_vmail_messages": int(record.get("Number_Vmail_Messages", 0)),
                "total_day_calls": int(record.get("Total_Day_Calls", 100)),
                "total_day_charge": float(record.get("Total_Day_Charge", 30.0)),
                "total_eve_calls": int(record.get("Total_Eve_Calls", 100)),
                "total_eve_charge": float(record.get("Total_Eve_Charge", 17.0)),
                "total_night_calls": int(record.get("Total_Night_Calls", 100)),
                "total_night_charge": float(record.get("Total_Night_Charge", 11.0)),
                "total_intl_calls": int(record.get("Total_Intl_Calls", 3)),
                "total_intl_charge": float(record.get("Total_Intl_Charge", 2.7)),
                "customer_service_calls": int(record.get("Customer_Service_Calls", 1)),
            })

        response = requests.post(
            f"{API_URL}/predict/batch",
            json={"customers": customers},
            timeout=30,
        )
        results = response.json()["predictions"]

        results_df = pd.DataFrame(results)
        output_df = pd.concat([df, results_df], axis=1)

        # Color-code the churn column
        st.dataframe(output_df, use_container_width=True)

        # Summary stats
        churn_count = results_df["churn"].sum()
        st.metric("Customers predicted to churn", f"{churn_count} / {len(results_df)}")

        # Download results
        csv = output_df.to_csv(index=False)
        st.download_button(
            "Download Results CSV",
            csv,
            file_name="churn_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
