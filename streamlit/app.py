# streamlit/app.py

"""
Streamlit dashboard for interactive churn predictions.

Calls the FastAPI backend for all predictions instead of loading
model artifacts directly.

"""

import os

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Hero subtitle */
.hero-subtitle {
    font-size: 1.2rem;
    color: #666;
    line-height: 1.6;
}

/* Card styling for goals, steps */
.card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 1.5rem;
    height: 100%;
    border: 1px solid #e9ecef;
}
.card-icon {
    font-size: 2.2rem;
    margin-bottom: 0.5rem;
}
.card h4 {
    margin: 0.3rem 0 0.5rem 0;
    color: #2c3e50;
}
.card p {
    color: #555;
    font-size: 0.95rem;
    line-height: 1.5;
    margin: 0;
}

/* Insight cards with colored left border */
.insight-card {
    border-left: 4px solid;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    background: #fafafa;
}
.insight-card h4 {
    margin: 0 0 0.4rem 0;
    color: #2c3e50;
    font-size: 1rem;
}
.insight-card p {
    margin: 0;
    color: #555;
    font-size: 0.92rem;
    line-height: 1.5;
}
.insight-card.risk {
    border-left-color: #e74c3c;
}
.insight-card.protective {
    border-left-color: #27ae60;
}
.insight-card.warning {
    border-left-color: #f39c12;
}

/* Prediction result banner */
.prediction-banner {
    padding: 1.2rem;
    border-radius: 10px;
    text-align: center;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.prediction-banner.churn {
    background: #fdecea;
    color: #c0392b;
    border: 2px solid #e74c3c;
}
.prediction-banner.no-churn {
    background: #eafaf1;
    color: #1e8449;
    border: 2px solid #27ae60;
}

/* Factor list containers */
.factor-box {
    border-radius: 8px;
    padding: 1rem 1.2rem;
}
.factor-box.risk {
    background: #fdf2f2;
    border: 1px solid #f5c6cb;
}
.factor-box.protective {
    background: #f0faf4;
    border: 1px solid #c3e6cb;
}
.factor-box h4 {
    margin: 0 0 0.6rem 0;
    font-size: 0.95rem;
}
.factor-box ul {
    margin: 0;
    padding-left: 1.2rem;
}
.factor-box li {
    color: #444;
    font-size: 0.92rem;
    margin-bottom: 0.3rem;
}
.factor-box li code {
    font-size: 0.82rem;
    color: #888;
}

/* Section headers */
.section-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #999;
    margin-bottom: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar: API status ─────────────────────────────────────────────────────

st.sidebar.title("System Status")

try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    model_loaded = health["model_loaded"]
    if model_loaded:
        st.sidebar.success("API connected &nbsp; | &nbsp; Model loaded")
    else:
        st.sidebar.warning("API connected &nbsp; | &nbsp; Model not loaded")
        st.sidebar.caption("Run `uv run train` to generate model artifacts.")
except requests.ConnectionError:
    model_loaded = False
    st.sidebar.error("API unavailable")
    st.sidebar.caption(
        f"Start the API first:\n\n"
        f"`uv run uvicorn api.main:app --port 8000`\n\n"
        f"Configured API URL: `{API_URL}`"
    )

st.sidebar.divider()
st.sidebar.caption(
    "**SyriaTel Churn Predictor** v0.2.0\n\n"
    "Powered by XGBoost + SHAP"
)

# ── Main content: Tabs ──────────────────────────────────────────────────────

tab_home, tab_single, tab_batch = st.tabs([
    "Home",
    "Single Prediction",
    "Batch Prediction",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: HOME
# ═════════════════════════════════════════════════════════════════════════════

with tab_home:
    st.title("SyriaTel Customer Churn Predictor")
    st.markdown(
        '<p class="hero-subtitle">'
        'An intelligent platform that predicts which customers are at risk of '
        'leaving SyriaTel, enabling proactive retention strategies before '
        "it's too late."
        '</p>',
        unsafe_allow_html=True,
    )

    st.divider()

    # Key metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Base Churn Rate", "14.5%")
    with m2:
        st.metric("Customers Analyzed", "3,333")
    with m3:
        st.metric("Model", "XGBoost")

    st.divider()

    # Platform goals
    st.subheader("What We Aim to Achieve")
    g1, g2, g3 = st.columns(3)

    with g1:
        st.markdown("""
        <div class="card">
            <div class="card-icon">🎯</div>
            <h4>Identify At-Risk Customers</h4>
            <p>Use machine learning to flag customers showing early signs of
            churn — before they make the decision to leave.</p>
        </div>
        """, unsafe_allow_html=True)
    with g2:
        st.markdown("""
        <div class="card">
            <div class="card-icon">🔍</div>
            <h4>Understand the Why</h4>
            <p>Every prediction comes with a breakdown of the factors driving
            it, so retention teams know exactly what to address.</p>
        </div>
        """, unsafe_allow_html=True)
    with g3:
        st.markdown("""
        <div class="card">
            <div class="card-icon">🛡️</div>
            <h4>Enable Proactive Retention</h4>
            <p>Armed with predictions and explanations, teams can target
            the right customers with the right interventions at the right time.</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # How it works
    st.subheader("How It Works")
    s1, s2, s3 = st.columns(3)

    with s1:
        st.markdown("""
        <div class="card">
            <div class="card-icon">📝</div>
            <h4>Step 1: Enter Customer Data</h4>
            <p>Provide usage details like call charges, service calls,
            and plan information — or upload a CSV for batch analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown("""
        <div class="card">
            <div class="card-icon">⚙️</div>
            <h4>Step 2: AI Model Analyzes</h4>
            <p>The XGBoost model processes the data through the same pipeline
            used during training, then SHAP computes per-feature contributions.</p>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown("""
        <div class="card">
            <div class="card-icon">📊</div>
            <h4>Step 3: Get Prediction + Explanation</h4>
            <p>Receive a churn probability, risk level, and a visual breakdown
            of which factors are driving the prediction.</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Key churn insights
    st.subheader("Key Churn Insights")
    st.caption("Based on analysis of 3,333 SyriaTel customers.")

    i1, i2 = st.columns(2)

    with i1:
        st.markdown("""
        <div class="insight-card risk">
            <h4>📞 Customer Service Calls</h4>
            <p>Customers with <strong>4+ service calls</strong> churn at <strong>51.7%</strong> —
            over 3x the baseline rate of 14.5%.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-card protective">
            <h4>📧 Voice Mail Plan</h4>
            <p>Voice mail subscribers churn at only <strong>8.7%</strong>,
            nearly half the rate of non-subscribers (16.7%).</p>
        </div>
        """, unsafe_allow_html=True)

    with i2:
        st.markdown("""
        <div class="insight-card risk">
            <h4>🌐 International Plan</h4>
            <p>International plan holders churn at <strong>42.4%</strong>,
            compared to just <strong>11.5%</strong> for those without.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-card warning">
            <h4>💰 High Day Charges</h4>
            <p>Customers in the top 25% of day charges churn at <strong>29.3%</strong>,
            over double the rate of bottom 25% spenders (12.2%).</p>
        </div>
        """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: SINGLE PREDICTION
# ═════════════════════════════════════════════════════════════════════════════

with tab_single:
    st.header("Single Customer Prediction")
    st.markdown("Enter a customer's usage data to predict their churn risk.")

    with st.form("prediction_form"):
        customer_name = st.text_input(
            "Customer Name (optional)",
            value="",
            help="For identification purposes only — not used by the model.",
        )

        st.divider()

        col_high, col_additional = st.columns(2)

        with col_high:
            st.subheader("High-Impact Features")

            customer_service_calls = st.number_input(
                "Customer Service Calls", min_value=0, max_value=20, value=1,
                help="Avg: 1.6. Values above 4 strongly indicate churn.",
            )
            total_day_charge = st.number_input(
                "Total Day Charge ($)", min_value=0.0, max_value=100.0, value=30.0, step=0.5,
                help="Avg: $30.56. Range: $0 - $75.90.",
            )
            international_plan = st.selectbox(
                "International Plan", ["no", "yes"],
                help="Customers with an international plan churn at a much higher rate.",
            )
            total_eve_charge = st.number_input(
                "Total Evening Charge ($)", min_value=0.0, max_value=50.0, value=17.0, step=0.5,
                help="Avg: $17.08. Range: $0 - $35.50.",
            )
            total_intl_charge = st.number_input(
                "Total International Charge ($)", min_value=0.0, max_value=10.0, value=2.70, step=0.1,
                help="Avg: $2.76. Range: $0 - $5.40.",
            )
            total_intl_calls = st.number_input(
                "Total International Calls", min_value=0, max_value=20, value=3,
                help="Avg: 4.5 calls. Range: 0 - 20.",
            )
            total_night_charge = st.number_input(
                "Total Night Charge ($)", min_value=0.0, max_value=50.0, value=11.0, step=0.5,
                help="Avg: $9.04. Range: $0 - $17.77.",
            )
            voice_mail_plan = st.selectbox(
                "Voice Mail Plan", ["no", "yes"],
                help="Customers with voicemail tend to churn less.",
            )

        with col_additional:
            st.subheader("Additional Features")

            state = st.text_input(
                "State (abbreviation)", value="OH", max_chars=2,
                help="Two-letter US state code, e.g. KS, OH, NJ.",
            )
            account_length = st.number_input(
                "Account Length (days)", min_value=0, max_value=500, value=100,
                help="Avg: 101 days. Range: 1 - 243.",
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
                help="Avg: 100 calls. Range: 0 - 165.",
            )
            total_eve_calls = st.number_input(
                "Total Evening Calls", min_value=0, max_value=200, value=100,
                help="Avg: 100 calls. Range: 0 - 170.",
            )
            total_night_calls = st.number_input(
                "Total Night Calls", min_value=0, max_value=200, value=100,
                help="Avg: 100 calls. Range: 0 - 175.",
            )

        submitted = st.form_submit_button("Predict Churn", use_container_width=True)

    # ── Prediction results ───────────────────────────────────────────────────

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

        st.divider()

        if customer_name.strip():
            st.subheader(f"Results for {customer_name.strip()}")

        # Prediction banner + metrics
        r1, r2, r3 = st.columns(3)

        with r1:
            if result["churn"]:
                st.markdown(
                    '<div class="prediction-banner churn">CHURN PREDICTED</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="prediction-banner no-churn">NO CHURN</div>',
                    unsafe_allow_html=True,
                )

        with r2:
            st.metric("Churn Probability", f"{result['churn_probability']:.1%}")

        with r3:
            risk = result["risk_level"]
            risk_icons = {"low": "🟢", "medium": "🟡", "high": "🔴"}
            st.metric("Risk Level", f"{risk_icons.get(risk, '')} {risk.upper()}")

        st.progress(
            result["churn_probability"],
            text=f"Churn probability: {result['churn_probability']:.1%}",
        )

        # SHAP feature contributions
        contributions = result.get("feature_contributions", [])
        if contributions:
            st.divider()
            st.subheader("What's driving this prediction?")

            # Bar chart
            features = [c["feature"] for c in reversed(contributions)]
            values = [c["contribution"] for c in reversed(contributions)]
            colors = ["#e74c3c" if v > 0 else "#27ae60" for v in values]

            fig = go.Figure(data=[
                go.Bar(
                    y=features,
                    x=values,
                    orientation="h",
                    marker_color=colors,
                )
            ])
            fig.update_layout(
                xaxis_title="Contribution to Churn Prediction",
                height=max(300, len(contributions) * 35),
                margin=dict(l=0, r=0, t=10, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Written explanation
            risk_factors = [c for c in contributions if c["contribution"] > 0]
            protective_factors = [c for c in contributions if c["contribution"] < 0]
            top_factor = contributions[0]["feature"]

            if result["churn"]:
                st.info(
                    f"The strongest factor driving this churn prediction is "
                    f"**{top_factor}**. "
                    f"There are **{len(risk_factors)} risk factor(s)** outweighing "
                    f"**{len(protective_factors)} protective factor(s)**.",
                    icon="🔍",
                )
            else:
                st.info(
                    f"This customer is unlikely to churn. The strongest signal is "
                    f"**{top_factor}**. "
                    f"There are **{len(protective_factors)} protective factor(s)** "
                    f"outweighing **{len(risk_factors)} risk factor(s)**.",
                    icon="🔍",
                )

            # Factor lists
            col_risk, col_protect = st.columns(2)

            with col_risk:
                risk_items = "".join(
                    f"<li>{f['feature']} <code>+{f['contribution']:.4f}</code></li>"
                    for f in risk_factors
                ) if risk_factors else "<li><em>None identified</em></li>"

                st.markdown(f"""
                <div class="factor-box risk">
                    <h4>⚠️ Risk Factors</h4>
                    <ul>{risk_items}</ul>
                </div>
                """, unsafe_allow_html=True)

            with col_protect:
                protect_items = "".join(
                    f"<li>{f['feature']} <code>{f['contribution']:.4f}</code></li>"
                    for f in protective_factors
                ) if protective_factors else "<li><em>None identified</em></li>"

                st.markdown(f"""
                <div class="factor-box protective">
                    <h4>✅ Protective Factors</h4>
                    <ul>{protect_items}</ul>
                </div>
                """, unsafe_allow_html=True)

        # Input summary
        with st.expander("View input data"):
            st.json(payload)

    elif submitted and not model_loaded:
        st.error("Cannot make predictions — the API model is not available.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: BATCH PREDICTION
# ═════════════════════════════════════════════════════════════════════════════

with tab_batch:
    st.header("Batch Prediction")
    st.markdown(
        "Upload a CSV file with customer data to predict churn for "
        "multiple customers at once."
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"Loaded **{len(df)} customers**")
        st.dataframe(df.head(), use_container_width=True)

        if not model_loaded:
            st.error("Cannot make predictions — the API model is not available.")
        elif st.button("Run Batch Prediction", use_container_width=True):
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

            st.divider()

            # Summary metrics
            churn_count = results_df["churn"].sum()
            bm1, bm2, bm3 = st.columns(3)
            with bm1:
                st.metric("Total Customers", len(results_df))
            with bm2:
                st.metric("Predicted to Churn", int(churn_count))
            with bm3:
                st.metric("Churn Rate", f"{churn_count / len(results_df):.1%}")

            # Results table
            st.dataframe(output_df, use_container_width=True)

            # Download
            csv = output_df.to_csv(index=False)
            st.download_button(
                "Download Results CSV",
                csv,
                file_name="churn_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
