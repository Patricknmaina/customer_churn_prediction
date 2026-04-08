"""
Streamlit dashboard for churn predictions and retention operations analytics.

The app calls the FastAPI backend for all predictions.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Required fields for the API single/batch prediction payloads.
REQUIRED_BATCH_COLUMNS: dict[str, list[str]] = {
    "customer_service_calls": ["Customer_Service_Calls", "customer_service_calls"],
    "total_day_charge": ["Total_Day_Charge", "total_day_charge"],
    "international_plan": ["International_Plan", "international_plan"],
    "total_eve_charge": ["Total_Eve_Charge", "total_eve_charge"],
    "total_intl_charge": ["Total_Intl_Charge", "total_intl_charge"],
    "total_intl_calls": ["Total_Intl_Calls", "total_intl_calls"],
    "total_night_charge": ["Total_Night_Charge", "total_night_charge"],
    "voice_mail_plan": ["Voice_Mail_Plan", "voice_mail_plan"],
}

OPTIONAL_BATCH_COLUMNS: dict[str, tuple[list[str], Any]] = {
    "state": (["State", "state"], "OH"),
    "account_length": (["Account_Length", "account_length"], 100),
    "area_code": (["Area_Code", "area_code"], 415),
    "number_vmail_messages": (["Number_Vmail_Messages", "number_vmail_messages"], 0),
    "total_day_calls": (["Total_Day_Calls", "total_day_calls"], 100),
    "total_eve_calls": (["Total_Eve_Calls", "total_eve_calls"], 100),
    "total_night_calls": (["Total_Night_Calls", "total_night_calls"], 100),
}

SEGMENT_COLUMNS = ["State", "Area_Code", "International_Plan", "Voice_Mail_Plan"]
RISK_ORDER = ["low", "medium", "high"]


# -----------------------------------------------------------------------------
# Page and theme
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="SyriaTel Retention Operations Console",
    page_icon=":material/insights:",
    layout="wide",
)

st.markdown(
    """
<style>
:root {
  --surface: #f4f7fb;
  --surface-card: #ffffff;
  --text-primary: #10233d;
  --text-secondary: #4d617a;
  --border: #d9e2ee;
  --brand: #0f4c81;
  --brand-soft: #eaf2fa;
  --risk-high: #b03a2e;
  --risk-medium: #a56a00;
  --risk-low: #1e7a4d;
}

.main .block-container {
  padding-top: 1.2rem;
  padding-bottom: 2rem;
}

.ops-card {
  background: var(--surface-card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1rem 1.1rem;
}

.ops-card h4 {
  margin: 0 0 0.35rem 0;
  color: var(--text-primary);
  font-size: 1.0rem;
}

.ops-card p {
  margin: 0;
  color: var(--text-secondary);
  font-size: 0.92rem;
  line-height: 1.4;
}

.kpi-row {
  border: 1px solid var(--border);
  background: var(--surface-card);
  border-radius: 12px;
  padding: 0.8rem 0.9rem;
}

.status-chip {
  border-radius: 999px;
  padding: 0.25rem 0.65rem;
  font-size: 0.78rem;
  font-weight: 600;
  display: inline-block;
}

.status-chip.ok {
  background: #e6f5ed;
  color: var(--risk-low);
}

.status-chip.warn {
  background: #fff3dd;
  color: var(--risk-medium);
}

.status-chip.err {
  background: #fce9e7;
  color: var(--risk-high);
}

.result-banner {
  border-radius: 12px;
  padding: 0.95rem 1rem;
  border: 1px solid var(--border);
  font-weight: 600;
}

.result-banner.low {
  background: #ecf8f1;
  color: var(--risk-low);
}

.result-banner.medium {
  background: #fff8e8;
  color: var(--risk-medium);
}

.result-banner.high {
  background: #fcebeb;
  color: var(--risk-high);
}

@media (max-width: 920px) {
  .main .block-container {
    padding-left: 0.8rem;
    padding-right: 0.8rem;
  }
}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# API helpers
# -----------------------------------------------------------------------------

def _safe_json(response: requests.Response) -> dict[str, Any] | None:
    try:
        data = response.json()
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def get_health_status() -> tuple[bool, bool, str]:
    """Return (api_connected, model_loaded, message)."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=6)
    except requests.RequestException as exc:
        return False, False, f"Unable to reach API: {exc}"

    if response.status_code != 200:
        return False, False, f"Health check failed with status {response.status_code}."

    data = _safe_json(response)
    if data is None:
        return False, False, "Health check returned invalid JSON."

    model_loaded = bool(data.get("model_loaded", False))
    return True, model_loaded, "API reachable and responsive."


def api_predict(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
    except requests.RequestException as exc:
        return None, f"Request failed: {exc}"

    data = _safe_json(response)
    if response.status_code != 200:
        detail = data.get("detail") if isinstance(data, dict) else response.text
        return None, f"Prediction failed (status {response.status_code}): {detail}"

    if data is None:
        return None, "Prediction response could not be parsed as JSON."

    return data, None


def api_predict_batch(customers: list[dict[str, Any]]) -> tuple[list[dict[str, Any]] | None, str | None]:
    try:
        response = requests.post(
            f"{API_URL}/predict/batch",
            json={"customers": customers},
            timeout=60,
        )
    except requests.RequestException as exc:
        return None, f"Batch request failed: {exc}"

    data = _safe_json(response)
    if response.status_code != 200:
        detail = data.get("detail") if isinstance(data, dict) else response.text
        return None, f"Batch prediction failed (status {response.status_code}): {detail}"

    if data is None or "predictions" not in data or not isinstance(data["predictions"], list):
        return None, "Batch response is missing a valid 'predictions' list."

    return data["predictions"], None


# -----------------------------------------------------------------------------
# Data and analytics helpers
# -----------------------------------------------------------------------------

def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    existing = {col.lower(): col for col in df.columns}
    for name in candidates:
        if name.lower() in existing:
            return existing[name.lower()]
    return None


def _coerce_row_value(value: Any, fallback: Any, *, numeric: bool = False, integer: bool = False) -> Any:
    if pd.isna(value):
        return fallback
    if numeric:
        parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(parsed):
            return fallback
        if integer:
            return int(parsed)
        return float(parsed)
    return str(value).strip() if isinstance(value, str) else value


def resolve_batch_column_mapping(
    df: pd.DataFrame,
) -> tuple[dict[str, str], dict[str, str | None], list[str], pd.DataFrame]:
    required_resolved: dict[str, str] = {}
    optional_resolved: dict[str, str | None] = {}
    missing: list[str] = []
    mapping_rows: list[dict[str, str]] = []

    for api_field, aliases in REQUIRED_BATCH_COLUMNS.items():
        col = _find_column(df, aliases)
        if col is None:
            missing.append(f"{api_field} (expected one of: {', '.join(aliases)})")
            mapping_rows.append(
                {
                    "API field": api_field,
                    "Required": "yes",
                    "Mapped source column": "NOT FOUND",
                }
            )
        else:
            required_resolved[api_field] = col
            mapping_rows.append(
                {
                    "API field": api_field,
                    "Required": "yes",
                    "Mapped source column": col,
                }
            )

    for api_field, (aliases, _default) in OPTIONAL_BATCH_COLUMNS.items():
        col = _find_column(df, aliases)
        optional_resolved[api_field] = col
        mapping_rows.append(
            {
                "API field": api_field,
                "Required": "no",
                "Mapped source column": col if col is not None else "DEFAULT",
            }
        )

    return required_resolved, optional_resolved, missing, pd.DataFrame(mapping_rows)


def build_batch_customers(
    df: pd.DataFrame, required_resolved: dict[str, str], optional_resolved: dict[str, str | None]
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    """Build API customers payload from uploaded DataFrame.

    Returns: (customers, valid_row_indices, invalid_csv_row_numbers)
    """
    customers: list[dict[str, Any]] = []
    valid_indices: list[int] = []
    invalid_rows: list[int] = []

    for idx, row in df.iterrows():
        try:
            customer = {
                "customer_service_calls": _coerce_row_value(
                    row[required_resolved["customer_service_calls"]], 1, numeric=True, integer=True
                ),
                "total_day_charge": _coerce_row_value(
                    row[required_resolved["total_day_charge"]], 30.0, numeric=True
                ),
                "international_plan": str(row[required_resolved["international_plan"]]).strip().lower(),
                "total_eve_charge": _coerce_row_value(
                    row[required_resolved["total_eve_charge"]], 17.0, numeric=True
                ),
                "total_intl_charge": _coerce_row_value(
                    row[required_resolved["total_intl_charge"]], 2.7, numeric=True
                ),
                "total_intl_calls": _coerce_row_value(
                    row[required_resolved["total_intl_calls"]], 3, numeric=True, integer=True
                ),
                "total_night_charge": _coerce_row_value(
                    row[required_resolved["total_night_charge"]], 11.0, numeric=True
                ),
                "voice_mail_plan": str(row[required_resolved["voice_mail_plan"]]).strip().lower(),
                "state": "OH",
                "account_length": 100,
                "area_code": 415,
                "number_vmail_messages": 0,
                "total_day_calls": 100,
                "total_eve_calls": 100,
                "total_night_calls": 100,
            }

            for api_field, (_aliases, default_value) in OPTIONAL_BATCH_COLUMNS.items():
                resolved = optional_resolved[api_field]
                if resolved is None:
                    continue

                if api_field in {"state"}:
                    customer[api_field] = _coerce_row_value(row[resolved], default_value)
                elif api_field in {"area_code", "number_vmail_messages", "total_day_calls", "total_eve_calls", "total_night_calls", "account_length"}:
                    customer[api_field] = _coerce_row_value(row[resolved], default_value, numeric=True, integer=True)

            if customer["international_plan"] not in {"yes", "no"}:
                raise ValueError("international_plan must be yes/no")
            if customer["voice_mail_plan"] not in {"yes", "no"}:
                raise ValueError("voice_mail_plan must be yes/no")

            customer["state"] = str(customer["state"]).upper()[:2]
            customers.append(customer)
            valid_indices.append(int(idx))
        except Exception:
            invalid_rows.append(int(idx) + 2)  # +2 for header + 1-indexed row number

    return customers, valid_indices, invalid_rows


def classify_recommended_action(risk_level: str, churn_probability: float) -> str:
    if risk_level == "high":
        return "Escalate to retention specialist within 24 hours"
    if risk_level == "medium" or churn_probability >= 0.45:
        return "Queue targeted follow-up offer this week"
    return "Monitor in standard lifecycle program"


def aggregate_feature_drivers(pred_df: pd.DataFrame) -> pd.DataFrame:
    agg: dict[str, dict[str, float]] = defaultdict(lambda: {"total": 0.0, "abs_total": 0.0, "count": 0})
    for contributions in pred_df.get("feature_contributions", pd.Series(dtype="object")):
        if not isinstance(contributions, list):
            continue
        for item in contributions:
            feature = item.get("feature")
            contrib = float(item.get("contribution", 0.0))
            if not feature:
                continue
            agg[feature]["total"] += contrib
            agg[feature]["abs_total"] += abs(contrib)
            agg[feature]["count"] += 1

    if not agg:
        return pd.DataFrame(columns=["feature", "total_contribution", "absolute_contribution", "mentions"])

    rows = [
        {
            "feature": feat,
            "total_contribution": vals["total"],
            "absolute_contribution": vals["abs_total"],
            "mentions": int(vals["count"]),
        }
        for feat, vals in agg.items()
    ]
    df = pd.DataFrame(rows)
    return df.sort_values("absolute_contribution", ascending=False)


def build_segment_breakdown(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    segment = (
        df.groupby(column_name, dropna=False)
        .agg(
            customers=("churn", "size"),
            predicted_churn_rate=("churn", "mean"),
            avg_probability=("churn_probability", "mean"),
            high_risk_count=("risk_level", lambda s: int((s == "high").sum())),
        )
        .reset_index()
        .sort_values("predicted_churn_rate", ascending=False)
    )
    segment["predicted_churn_rate"] = (segment["predicted_churn_rate"] * 100).round(2)
    segment["avg_probability"] = (segment["avg_probability"] * 100).round(2)
    return segment


# -----------------------------------------------------------------------------
# Rendering helpers
# -----------------------------------------------------------------------------

def render_sidebar_status(api_ok: bool, model_loaded: bool, health_message: str) -> str:
    st.sidebar.title("Operations Control")
    if api_ok and model_loaded:
        st.sidebar.markdown('<span class="status-chip ok">API CONNECTED | MODEL READY</span>', unsafe_allow_html=True)
    elif api_ok and not model_loaded:
        st.sidebar.markdown('<span class="status-chip warn">API CONNECTED | MODEL NOT READY</span>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<span class="status-chip err">API UNAVAILABLE</span>', unsafe_allow_html=True)

    st.sidebar.caption(health_message)
    st.sidebar.caption(f"Endpoint: `{API_URL}`")
    st.sidebar.divider()

    if "workspace" not in st.session_state:
        st.session_state["workspace"] = "Overview"

    return st.sidebar.radio(
        "Workspace",
        ["Overview", "Single Prediction", "Batch Operations"],
        key="workspace",
    )


def render_overview(api_ok: bool, model_loaded: bool) -> None:
    st.title("SyriaTel Retention Operations Console")
    st.caption("Operational decision support for customer churn mitigation.")

    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("Base churn rate", "14.5%")
    with r2:
        st.metric("Model", "XGBoost")
    with r3:
        readiness = "Ready" if api_ok and model_loaded else "Attention Needed"
        st.metric("System readiness", readiness)

    st.markdown("### Readiness")
    readiness_text = (
        "System is ready for live triage and batch scoring."
        if api_ok and model_loaded
        else "Prediction service is not fully ready. Resolve API/model state before operations."
    )
    st.markdown(
        f"""
<div class="ops-card">
  <h4>Current service state</h4>
  <p>{readiness_text}</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("### Quick actions")
    q1, q2 = st.columns(2)
    with q1:
        if st.button("Open single prediction workflow", use_container_width=True):
            st.session_state["workspace"] = "Single Prediction"
            st.rerun()
    with q2:
        if st.button("Open batch operations workflow", use_container_width=True):
            st.session_state["workspace"] = "Batch Operations"
            st.rerun()

    st.markdown("### Model usage guidance")
    g1, g2 = st.columns(2)
    with g1:
        st.markdown(
            """
<div class="ops-card">
  <h4>High-impact drivers</h4>
  <p>Customer service calls, day charge, international plan, and voice mail plan have strong influence on risk outcomes.</p>
</div>
""",
            unsafe_allow_html=True,
        )
    with g2:
        st.markdown(
            """
<div class="ops-card">
  <h4>Operational note</h4>
  <p>Use batch analytics to prioritize high-risk workload and allocate retention resources by segment.</p>
</div>
""",
            unsafe_allow_html=True,
        )


def render_single_prediction(api_ok: bool, model_loaded: bool) -> None:
    st.header("Single customer triage")
    st.caption("High-impact fields are required; advanced fields are optional and prefilled with safe defaults.")

    with st.form("single_prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Required inputs")
            customer_service_calls = st.number_input("Customer Service Calls", min_value=0, max_value=20, value=1)
            total_day_charge = st.number_input("Total Day Charge (USD)", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
            international_plan = st.selectbox("International Plan", ["no", "yes"], index=0)
            total_eve_charge = st.number_input("Total Evening Charge (USD)", min_value=0.0, max_value=50.0, value=17.0, step=0.1)
            total_intl_charge = st.number_input("Total International Charge (USD)", min_value=0.0, max_value=10.0, value=2.7, step=0.1)
            total_intl_calls = st.number_input("Total International Calls", min_value=0, max_value=30, value=3)
            total_night_charge = st.number_input("Total Night Charge (USD)", min_value=0.0, max_value=50.0, value=11.0, step=0.1)
            voice_mail_plan = st.selectbox("Voice Mail Plan", ["no", "yes"], index=0)

        with c2:
            with st.expander("Advanced inputs", expanded=False):
                state = st.text_input("State", value="OH", max_chars=2, help="Two-letter US state code")
                account_length = st.number_input("Account Length (days)", min_value=0, max_value=500, value=100)
                area_code = st.selectbox("Area Code", [408, 415, 510], index=1)
                number_vmail_messages = st.number_input("Number of Voicemail Messages", min_value=0, max_value=80, value=0)
                total_day_calls = st.number_input("Total Day Calls", min_value=0, max_value=250, value=100)
                total_eve_calls = st.number_input("Total Evening Calls", min_value=0, max_value=250, value=100)
                total_night_calls = st.number_input("Total Night Calls", min_value=0, max_value=250, value=100)

        submitted = st.form_submit_button("Run prediction", use_container_width=True)

    if not submitted:
        return

    if not api_ok:
        st.error("API is unavailable. Verify backend connectivity before scoring.")
        return
    if not model_loaded:
        st.error("Model is not loaded on the backend. Check service startup logs.")
        return

    if len(state.strip()) != 2:
        st.error("State must be a 2-character abbreviation.")
        return

    payload = {
        "state": state.strip().upper(),
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

    with st.spinner("Scoring customer profile..."):
        result, error = api_predict(payload)

    if error:
        st.error(error)
        return

    churn_probability = float(result.get("churn_probability", 0.0))
    risk_level = str(result.get("risk_level", "low"))
    churn = bool(result.get("churn", False))
    contributions = result.get("feature_contributions", [])

    st.markdown("### Prediction outcome")
    outcome_text = "Customer flagged as likely to churn" if churn else "Customer not flagged for churn"
    st.markdown(
        f'<div class="result-banner {risk_level}">{outcome_text}</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Churn probability", f"{churn_probability:.1%}")
    with m2:
        st.metric("Risk level", risk_level.title())
    with m3:
        st.metric("Recommended action", classify_recommended_action(risk_level, churn_probability))

    st.progress(churn_probability, text=f"Probability confidence: {churn_probability:.1%}")

    st.markdown("### Driver analysis")
    if not contributions:
        st.info("No feature contribution breakdown returned by the API for this prediction.")
    else:
        contrib_df = pd.DataFrame(contributions)
        contrib_df["direction"] = contrib_df["contribution"].apply(lambda x: "Risk Up" if x > 0 else "Protective")
        contrib_df = contrib_df.sort_values("contribution", key=lambda s: s.abs(), ascending=False)

        fig = go.Figure(
            data=[
                go.Bar(
                    y=list(reversed(contrib_df["feature"].tolist())),
                    x=list(reversed(contrib_df["contribution"].tolist())),
                    orientation="h",
                    marker_color=["#b03a2e" if v > 0 else "#1e7a4d" for v in reversed(contrib_df["contribution"].tolist())],
                )
            ]
        )
        fig.update_layout(
            xaxis_title="Contribution to churn prediction",
            yaxis_title="",
            height=max(320, len(contrib_df) * 38),
            margin=dict(l=0, r=0, t=10, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.caption("Top risk-increasing drivers")
            st.dataframe(
                contrib_df[contrib_df["contribution"] > 0][["feature", "contribution"]].head(5),
                use_container_width=True,
                hide_index=True,
            )
        with c2:
            st.caption("Top protective drivers")
            st.dataframe(
                contrib_df[contrib_df["contribution"] < 0][["feature", "contribution"]].head(5),
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("View submitted payload"):
        st.json(payload)


def render_batch_operations(api_ok: bool, model_loaded: bool) -> None:
    st.header("Batch operations")
    st.caption("Upload customer records, run scoring, and operationalize retention prioritization.")

    st.markdown("### Input requirements")
    req_df = pd.DataFrame(
        {
            "API field": list(REQUIRED_BATCH_COLUMNS.keys()),
            "Accepted source column names": [", ".join(v) for v in REQUIRED_BATCH_COLUMNS.values()],
        }
    )
    st.dataframe(req_df, use_container_width=True, hide_index=True)

    template_df = pd.DataFrame(
        [
            {
                "State": "OH",
                "Account_Length": 100,
                "Area_Code": 415,
                "International_Plan": "no",
                "Voice_Mail_Plan": "yes",
                "Number_Vmail_Messages": 25,
                "Total_Day_Calls": 110,
                "Total_Day_Charge": 45.07,
                "Total_Eve_Calls": 99,
                "Total_Eve_Charge": 16.78,
                "Total_Night_Calls": 91,
                "Total_Night_Charge": 11.01,
                "Total_Intl_Calls": 3,
                "Total_Intl_Charge": 2.70,
                "Customer_Service_Calls": 1,
            }
        ]
    )
    st.download_button(
        "Download CSV template",
        data=template_df.to_csv(index=False),
        file_name="batch_input_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])

    if uploaded is None:
        return

    try:
        source_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read uploaded CSV: {exc}")
        return

    if source_df.empty:
        st.warning("Uploaded file is empty. Please provide at least one customer row.")
        return

    st.caption(f"Loaded rows: {len(source_df)}")
    st.dataframe(source_df.head(10), use_container_width=True)

    required_resolved, optional_resolved, missing_required, mapping_df = resolve_batch_column_mapping(source_df)
    st.caption("Column mapping summary")
    st.dataframe(mapping_df, use_container_width=True, hide_index=True)

    if missing_required:
        st.error("Missing required columns for batch scoring:")
        for msg in missing_required:
            st.write(f"- {msg}")
        return

    customers, valid_indices, invalid_rows = build_batch_customers(source_df, required_resolved, optional_resolved)

    if invalid_rows:
        st.warning(
            "Some rows contain invalid values and will be excluded from scoring. "
            f"CSV row numbers: {', '.join(map(str, invalid_rows[:20]))}"
        )

    st.info(f"Rows prepared for scoring: {len(customers)} of {len(source_df)}")

    if len(customers) == 0:
        st.error("No valid rows are available for scoring after validation.")
        return

    if not api_ok:
        st.error("API is unavailable. Verify backend connectivity before batch scoring.")
        return
    if not model_loaded:
        st.error("Model is not loaded on the backend. Check service startup logs.")
        return

    if st.button("Run batch prediction", use_container_width=True):
        with st.spinner("Running batch inference and generating analytics..."):
            predictions, error = api_predict_batch(customers)

        if error:
            st.error(error)
            return

        if len(predictions) != len(customers):
            st.error(
                "Batch response size mismatch. "
                f"Expected {len(customers)} predictions but received {len(predictions)}."
            )
            return

        working_df = source_df.copy().iloc[valid_indices].reset_index(drop=True)
        pred_df = pd.DataFrame(predictions)
        pred_df["recommended_action"] = pred_df.apply(
            lambda row: classify_recommended_action(str(row.get("risk_level", "low")), float(row.get("churn_probability", 0.0))),
            axis=1,
        )

        output_df = pd.concat([working_df, pred_df], axis=1)
        st.session_state["batch_output_df"] = output_df

    output_df = st.session_state.get("batch_output_df")
    if output_df is None or not isinstance(output_df, pd.DataFrame) or output_df.empty:
        return

    st.markdown("### Results controls")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        risk_filter = st.multiselect("Risk levels", options=RISK_ORDER, default=RISK_ORDER)
    with c2:
        churn_filter = st.selectbox("Predicted class", ["All", "Churn only", "No churn only"], index=0)
    with c3:
        min_probability = st.slider("Minimum probability", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    with c4:
        sort_field = st.selectbox(
            "Sort by",
            ["churn_probability", "risk_level", "churn"],
            index=0,
        )

    filtered = output_df.copy()
    if risk_filter:
        filtered = filtered[filtered["risk_level"].isin(risk_filter)]
    if churn_filter == "Churn only":
        filtered = filtered[filtered["churn"] == True]
    elif churn_filter == "No churn only":
        filtered = filtered[filtered["churn"] == False]
    filtered = filtered[filtered["churn_probability"] >= min_probability]

    if sort_field == "risk_level":
        filtered["risk_level"] = pd.Categorical(filtered["risk_level"], categories=RISK_ORDER, ordered=True)
        filtered = filtered.sort_values(["risk_level", "churn_probability"], ascending=[False, False])
        filtered["risk_level"] = filtered["risk_level"].astype(str)
    else:
        filtered = filtered.sort_values(sort_field, ascending=False)

    if filtered.empty:
        st.warning("No rows match the selected filters. Adjust filters to view analytics.")
        return

    st.markdown("### Triage KPIs")
    total_customers = len(filtered)
    churn_count = int(filtered["churn"].sum()) if total_customers else 0
    avg_probability = float(filtered["churn_probability"].mean()) if total_customers else 0.0
    high_risk_count = int((filtered["risk_level"] == "high").sum()) if total_customers else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Total customers", total_customers)
    with k2:
        st.metric("Predicted churn", churn_count)
    with k3:
        churn_rate = (churn_count / total_customers) if total_customers else 0.0
        st.metric("Predicted churn rate", f"{churn_rate:.1%}")
    with k4:
        st.metric("Average probability", f"{avg_probability:.1%}")
    with k5:
        st.metric("High-risk workload", high_risk_count)

    st.markdown("### Operational analytics")
    a1, a2 = st.columns(2)

    with a1:
        risk_counts = filtered["risk_level"].value_counts().reindex(RISK_ORDER, fill_value=0)
        risk_fig = go.Figure(
            data=[
                go.Bar(
                    x=risk_counts.index.tolist(),
                    y=risk_counts.values.tolist(),
                    marker_color=["#1e7a4d", "#a56a00", "#b03a2e"],
                )
            ]
        )
        risk_fig.update_layout(title="Risk distribution", xaxis_title="Risk level", yaxis_title="Customers", height=340)
        st.plotly_chart(risk_fig, use_container_width=True)

    with a2:
        prob_fig = go.Figure(data=[go.Histogram(x=filtered["churn_probability"], nbinsx=20, marker_color="#0f4c81")])
        prob_fig.update_layout(
            title="Churn probability distribution",
            xaxis_title="Probability",
            yaxis_title="Customer count",
            height=340,
        )
        st.plotly_chart(prob_fig, use_container_width=True)

    st.markdown("### Top predicted churn customers")
    top_n = st.slider("Rows to display", min_value=5, max_value=100, value=20, step=5)
    top_customers = filtered.sort_values("churn_probability", ascending=False).head(top_n)
    st.dataframe(top_customers, use_container_width=True, hide_index=True)

    st.markdown("### Segment breakdowns")
    available_segments = [col for col in SEGMENT_COLUMNS if col in filtered.columns]
    if not available_segments:
        st.info("No standard segment columns found in uploaded file for segment-level analytics.")
    else:
        selected_segment = st.selectbox("Segment by", available_segments)
        segment_df = build_segment_breakdown(filtered, selected_segment)
        st.dataframe(segment_df, use_container_width=True, hide_index=True)

    st.markdown("### Feature-driver summary")
    high_risk_subset = filtered[filtered["risk_level"] == "high"]
    driver_source = high_risk_subset if not high_risk_subset.empty else filtered
    drivers = aggregate_feature_drivers(driver_source)
    if drivers.empty:
        st.info("Feature contribution data was not present in results.")
    else:
        top_drivers = drivers.head(12)
        driver_fig = go.Figure(
            data=[
                go.Bar(
                    y=list(reversed(top_drivers["feature"].tolist())),
                    x=list(reversed(top_drivers["total_contribution"].tolist())),
                    orientation="h",
                    marker_color=["#b03a2e" if v > 0 else "#1e7a4d" for v in reversed(top_drivers["total_contribution"].tolist())],
                )
            ]
        )
        driver_fig.update_layout(
            title="Aggregated top drivers",
            xaxis_title="Net contribution",
            yaxis_title="",
            height=max(340, len(top_drivers) * 32),
        )
        st.plotly_chart(driver_fig, use_container_width=True)
        st.dataframe(top_drivers, use_container_width=True, hide_index=True)

    st.markdown("### Exports")
    exp1, exp2 = st.columns(2)
    with exp1:
        st.download_button(
            "Download filtered results",
            data=filtered.to_csv(index=False),
            file_name="filtered_churn_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with exp2:
        high_risk_export = filtered[filtered["risk_level"] == "high"]
        st.download_button(
            "Download high-risk workload",
            data=high_risk_export.to_csv(index=False),
            file_name="high_risk_customers.csv",
            mime="text/csv",
            use_container_width=True,
        )


# -----------------------------------------------------------------------------
# Main app routing
# -----------------------------------------------------------------------------

api_ok, model_loaded, health_message = get_health_status()
workspace = render_sidebar_status(api_ok, model_loaded, health_message)

if workspace == "Overview":
    render_overview(api_ok, model_loaded)
elif workspace == "Single Prediction":
    render_single_prediction(api_ok, model_loaded)
else:
    render_batch_operations(api_ok, model_loaded)
