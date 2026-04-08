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

RISK_COLORS = {
    "low": "#1C7A50",
    "medium": "#9A6A12",
    "high": "#B5403B",
}

PLOT_COLORS = ["#145DA0", "#3C91E6", "#9BC53D", "#F29E4C", "#B5403B"]


# -----------------------------------------------------------------------------
# Page and design system
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
  --color-brand-700: #123E63;
  --color-brand-600: #145DA0;
  --color-brand-100: #E9F2FB;
  --color-surface-0: #FFFFFF;
  --color-surface-1: #F5F8FC;
  --color-surface-2: #EEF3FA;
  --color-border: #D7E0EA;
  --color-text-strong: #0F2438;
  --color-text-body: #3E5266;
  --color-text-muted: #6C8093;
  --color-success-bg: #EAF7F0;
  --color-success-fg: #1C7A50;
  --color-warning-bg: #FFF5E7;
  --color-warning-fg: #9A6A12;
  --color-error-bg: #FDEEEE;
  --color-error-fg: #B5403B;
  --color-info-bg: #EAF2FB;
  --color-info-fg: #145DA0;

  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 16px;

  --space-1: 0.25rem;
  --space-2: 0.5rem;
  --space-3: 0.75rem;
  --space-4: 1rem;
  --space-5: 1.25rem;
  --space-6: 1.5rem;

  --font-display: 1.55rem;
  --font-h2: 1.2rem;
  --font-h3: 1.02rem;
  --font-body: 0.95rem;
  --font-caption: 0.84rem;

  --line-body: 1.5;
}

.main .block-container {
  max-width: 1260px;
  padding-top: 1.1rem;
  padding-bottom: 2.2rem;
}

.layout-shell {
  background: #ffffff;
  border: 1px solid #d5e1ee;
  border-radius: var(--radius-lg);
  padding: var(--space-5) var(--space-6) var(--space-5) var(--space-6);
  margin-bottom: var(--space-5);
  box-shadow: 0 4px 20px rgba(18, 62, 99, 0.06);
  position: relative;
}

.layout-shell:before {
  content: "";
  position: absolute;
  left: 0;
  top: 0;
  width: 100%;
  height: 5px;
  background: linear-gradient(90deg, var(--color-brand-600), #2d7cd3);
  border-radius: var(--radius-lg) var(--radius-lg) 0 0;
}

.layout-header {
  font-size: var(--font-display);
  line-height: 1.2;
  color: var(--color-text-strong);
  margin: 0;
  font-weight: 700;
}

.layout-subtitle {
  margin: var(--space-2) 0 0 0;
  color: var(--color-text-body);
  font-size: var(--font-body);
  line-height: var(--line-body);
}

.section-title {
  font-size: var(--font-h2);
  margin: var(--space-5) 0 var(--space-3) 0;
  color: var(--color-text-strong);
  font-weight: 650;
}

.card-panel {
  background: #ffffff;
  border: 1px solid #dbe4ef;
  border-radius: var(--radius-md);
  padding: var(--space-4) var(--space-5);
  box-shadow: 0 1px 4px rgba(15, 36, 56, 0.03);
}

.card-panel h4 {
  margin: 0 0 var(--space-2) 0;
  color: var(--color-text-strong);
  font-size: var(--font-h3);
}

.card-panel p {
  margin: 0;
  color: var(--color-text-body);
  font-size: var(--font-body);
  line-height: var(--line-body);
}

.card-kpi {
  background: #ffffff;
  border: 1px solid #dbe4ef;
  border-radius: var(--radius-md);
  padding: var(--space-3) var(--space-4);
  min-height: 82px;
}

.card-kpi .kpi-label {
  font-size: var(--font-caption);
  color: var(--color-text-muted);
  margin-bottom: var(--space-1);
}

.card-kpi .kpi-value {
  font-size: 1.3rem;
  font-weight: 700;
  color: var(--color-text-strong);
}

.status-chip {
  border-radius: 999px;
  padding: 0.25rem 0.72rem;
  font-size: 0.76rem;
  font-weight: 700;
  letter-spacing: 0.02em;
  display: inline-block;
}

.status-ok {
  background: var(--color-success-bg);
  color: var(--color-success-fg);
}

.status-warn {
  background: var(--color-warning-bg);
  color: var(--color-warning-fg);
}

.status-err {
  background: var(--color-error-bg);
  color: var(--color-error-fg);
}

.alert-box {
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  padding: var(--space-3) var(--space-4);
  margin: var(--space-2) 0;
  font-size: var(--font-body);
}

.alert-info {
  background: var(--color-info-bg);
  color: var(--color-info-fg);
}

.alert-warning {
  background: var(--color-warning-bg);
  color: var(--color-warning-fg);
}

.alert-error {
  background: var(--color-error-bg);
  color: var(--color-error-fg);
}

.result-banner {
  border-radius: var(--radius-md);
  border: 1px solid var(--color-border);
  padding: var(--space-4);
  font-size: var(--font-h3);
  font-weight: 650;
}

.result-low {
  background: var(--color-success-bg);
  color: var(--color-success-fg);
}

.result-medium {
  background: var(--color-warning-bg);
  color: var(--color-warning-fg);
}

.result-high {
  background: var(--color-error-bg);
  color: var(--color-error-fg);
}

.table-hint {
  color: var(--color-text-muted);
  font-size: var(--font-caption);
  margin-top: var(--space-1);
}

div[data-testid="stDataFrame"] {
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  overflow: hidden;
  background: #ffffff;
}

div[data-testid="stMetric"] {
  background: #ffffff;
  border: 1px solid #dbe4ef;
  border-radius: var(--radius-sm);
  padding: 0.5rem 0.7rem;
}

div[data-testid="stForm"] {
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  padding: var(--space-4);
  background: var(--color-surface-0);
}

div[data-testid="stExpander"] {
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
}

.stButton > button, .stDownloadButton > button {
  border-radius: var(--radius-sm);
  border: 1px solid #b8cce3;
  background: #ffffff;
  color: var(--color-brand-700);
  transition: transform 0.16s ease, box-shadow 0.16s ease;
  font-weight: 600;
}

.stButton > button:hover, .stDownloadButton > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(20, 93, 160, 0.14);
  border-color: #8fb2d8;
}

.stButton > button:focus-visible,
.stDownloadButton > button:focus-visible {
  outline: 3px solid rgba(20, 93, 160, 0.33);
  outline-offset: 2px;
}

hr {
  margin-top: 0.8rem;
  margin-bottom: 0.8rem;
  border-color: var(--color-border);
}

div[data-testid="stTabs"] [data-baseweb="tab-list"] {
  gap: 8px;
}

div[data-testid="stTabs"] button[role="tab"] {
  border: 1px solid #d4dfec;
  border-radius: 10px;
  padding: 0.3rem 0.8rem;
  background: #ffffff;
}

div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  border-color: #8fb2d8;
  background: #edf5fd;
  color: #0f3f68;
}

@media (max-width: 980px) {
  .main .block-container {
    padding-left: 0.75rem;
    padding-right: 0.75rem;
  }

  .layout-shell {
    padding: var(--space-4);
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

                if api_field == "state":
                    customer[api_field] = _coerce_row_value(row[resolved], default_value)
                elif api_field in {
                    "area_code",
                    "number_vmail_messages",
                    "total_day_calls",
                    "total_eve_calls",
                    "total_night_calls",
                    "account_length",
                }:
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


def apply_plotly_theme(fig: go.Figure, *, height: int = 340, title: str = "") -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        colorway=PLOT_COLORS,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font={"family": "ui-sans-serif, system-ui, -apple-system, Segoe UI, Arial", "size": 13, "color": "#1C334A"},
        margin={"l": 6, "r": 6, "t": 42, "b": 22},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        hoverlabel={"font_size": 12},
    )
    fig.update_xaxes(gridcolor="#E8EEF5", zerolinecolor="#DCE5EF")
    fig.update_yaxes(gridcolor="#E8EEF5", zerolinecolor="#DCE5EF")
    return fig


# -----------------------------------------------------------------------------
# Rendering helpers
# -----------------------------------------------------------------------------

def render_page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
<div class="layout-shell">
  <h1 class="layout-header">{title}</h1>
  <p class="layout-subtitle">{subtitle}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_panel(title: str, body: str) -> None:
    st.markdown(
        f"""
<div class="card-panel">
  <h4>{title}</h4>
  <p>{body}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_alert(kind: str, text: str) -> None:
    class_name = {
        "info": "alert-info",
        "warning": "alert-warning",
        "error": "alert-error",
    }.get(kind, "alert-info")
    st.markdown(f'<div class="alert-box {class_name}">{text}</div>', unsafe_allow_html=True)


def render_kpi_card(label: str, value: str) -> None:
    st.markdown(
        f"""
<div class="card-kpi">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">{value}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_sidebar_status(api_ok: bool, model_loaded: bool, health_message: str) -> str:
    st.sidebar.markdown("### Operations Control")
    if api_ok and model_loaded:
        st.sidebar.markdown('<span class="status-chip status-ok">API CONNECTED | MODEL READY</span>', unsafe_allow_html=True)
    elif api_ok and not model_loaded:
        st.sidebar.markdown('<span class="status-chip status-warn">API CONNECTED | MODEL NOT READY</span>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<span class="status-chip status-err">API UNAVAILABLE</span>', unsafe_allow_html=True)

    st.sidebar.caption(health_message)
    st.sidebar.caption(f"Endpoint: `{API_URL}`")
    st.sidebar.divider()

    options = ["Overview", "Single Prediction", "Batch Operations"]
    current_workspace = st.session_state.get("workspace", "Overview")
    if current_workspace not in options:
        current_workspace = "Overview"

    st.sidebar.caption("Workspace")
    selected_workspace = st.sidebar.radio(
        "Workspace",
        options,
        index=options.index(current_workspace),
        label_visibility="collapsed",
    )
    st.session_state["workspace"] = selected_workspace
    return selected_workspace


def render_overview(api_ok: bool, model_loaded: bool) -> None:
    render_page_header(
        "SyriaTel Retention Operations Console",
        "Operator workspace for churn triage, scoring, and retention prioritization.",
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        render_kpi_card("Base churn rate", "14.5%")
    with m2:
        render_kpi_card("Model", "XGBoost")
    with m3:
        readiness = "Ready" if api_ok and model_loaded else "Attention Needed"
        render_kpi_card("System readiness", readiness)

    st.markdown('<h3 class="section-title">Readiness and actions</h3>', unsafe_allow_html=True)
    readiness_text = (
        "Service and model are available for live triage and batch scoring."
        if api_ok and model_loaded
        else "Service is reachable but not fully operational. Resolve backend/model readiness before production use."
    )
    c1, c2 = st.columns([1.4, 1])
    with c1:
        render_panel("Current service state", readiness_text)
    with c2:
        render_panel(
            "Operational guidance",
            "Use batch scoring for workload prioritization and single-customer triage for real-time intervention.",
        )

    a1, a2 = st.columns(2)
    with a1:
        if st.button("Open single prediction", type="primary", use_container_width=True):
            st.session_state["workspace"] = "Single Prediction"
            st.rerun()
    with a2:
        if st.button("Open batch operations", use_container_width=True):
            st.session_state["workspace"] = "Batch Operations"
            st.rerun()


def render_single_prediction(api_ok: bool, model_loaded: bool) -> None:
    render_page_header(
        "Single Customer Triage",
        "Score one customer profile, review risk level, and identify high-impact drivers.",
    )

    with st.form("single_prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Required fields")
            customer_service_calls = st.number_input("Customer Service Calls", min_value=0, max_value=20, value=1)
            total_day_charge = st.number_input("Total Day Charge (USD)", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
            international_plan = st.selectbox("International Plan", ["no", "yes"], index=0)
            total_eve_charge = st.number_input("Total Evening Charge (USD)", min_value=0.0, max_value=50.0, value=17.0, step=0.1)
            total_intl_charge = st.number_input("Total International Charge (USD)", min_value=0.0, max_value=10.0, value=2.7, step=0.1)
            total_intl_calls = st.number_input("Total International Calls", min_value=0, max_value=30, value=3)
            total_night_charge = st.number_input("Total Night Charge (USD)", min_value=0.0, max_value=50.0, value=11.0, step=0.1)
            voice_mail_plan = st.selectbox("Voice Mail Plan", ["no", "yes"], index=0)

        with c2:
            with st.expander("Advanced fields", expanded=False):
                st.caption("Optional operational fields with safe defaults")
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
        render_alert("error", "API is unavailable. Verify backend connectivity before scoring.")
        return
    if not model_loaded:
        render_alert("error", "Model is not loaded on the backend. Check service startup logs.")
        return

    if len(state.strip()) != 2:
        render_alert("warning", "State must be a 2-character abbreviation.")
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
        render_alert("error", error)
        return

    churn_probability = float(result.get("churn_probability", 0.0))
    risk_level = str(result.get("risk_level", "low"))
    churn = bool(result.get("churn", False))
    contributions = result.get("feature_contributions", [])

    st.markdown('<h3 class="section-title">Prediction outcome</h3>', unsafe_allow_html=True)
    outcome_text = "Customer flagged as likely to churn" if churn else "Customer not flagged for churn"
    st.markdown(
        f'<div class="result-banner result-{risk_level}">{outcome_text}</div>',
        unsafe_allow_html=True,
    )

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Churn probability", f"{churn_probability:.1%}")
    with k2:
        st.metric("Risk level", risk_level.title())
    with k3:
        st.metric("Recommended action", classify_recommended_action(risk_level, churn_probability))

    st.progress(churn_probability, text=f"Probability confidence: {churn_probability:.1%}")

    st.markdown('<h3 class="section-title">Driver analysis</h3>', unsafe_allow_html=True)
    if not contributions:
        render_alert("info", "No feature contribution breakdown returned by the API for this prediction.")
    else:
        contrib_df = pd.DataFrame(contributions)
        contrib_df = contrib_df.sort_values("contribution", key=lambda s: s.abs(), ascending=False)

        fig = go.Figure(
            data=[
                go.Bar(
                    y=list(reversed(contrib_df["feature"].tolist())),
                    x=list(reversed(contrib_df["contribution"].tolist())),
                    orientation="h",
                    marker_color=[
                        RISK_COLORS["high"] if v > 0 else RISK_COLORS["low"]
                        for v in reversed(contrib_df["contribution"].tolist())
                    ],
                )
            ]
        )
        fig = apply_plotly_theme(fig, height=max(320, len(contrib_df) * 38), title="Feature contribution impact")
        fig.update_layout(xaxis_title="Contribution to churn prediction", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        d1, d2 = st.columns(2)
        with d1:
            st.caption("Top risk-increasing drivers")
            st.dataframe(
                contrib_df[contrib_df["contribution"] > 0][["feature", "contribution"]].head(5),
                use_container_width=True,
                hide_index=True,
            )
        with d2:
            st.caption("Top protective drivers")
            st.dataframe(
                contrib_df[contrib_df["contribution"] < 0][["feature", "contribution"]].head(5),
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("View submitted payload"):
        st.json(payload)


def render_batch_operations(api_ok: bool, model_loaded: bool) -> None:
    render_page_header(
        "Batch Operations",
        "Upload customer data, run bulk scoring, and operationalize retention actions.",
    )

    with st.expander("Input schema and template", expanded=True):
        req_df = pd.DataFrame(
            {
                "API field": list(REQUIRED_BATCH_COLUMNS.keys()),
                "Accepted source column names": [", ".join(v) for v in REQUIRED_BATCH_COLUMNS.values()],
            }
        )
        st.dataframe(req_df, use_container_width=True, hide_index=True)
        st.markdown('<div class="table-hint">Required columns are validated before scoring.</div>', unsafe_allow_html=True)

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
        render_alert("error", f"Could not read uploaded CSV: {exc}")
        return

    if source_df.empty:
        render_alert("warning", "Uploaded file is empty. Please provide at least one customer row.")
        return

    st.caption(f"Loaded rows: {len(source_df)}")
    st.dataframe(source_df.head(10), use_container_width=True)

    required_resolved, optional_resolved, missing_required, mapping_df = resolve_batch_column_mapping(source_df)
    st.markdown("#### Column mapping summary")
    st.dataframe(mapping_df, use_container_width=True, hide_index=True)

    if missing_required:
        render_alert("error", "Missing required columns for batch scoring:")
        for msg in missing_required:
            st.write(f"- {msg}")
        return

    customers, valid_indices, invalid_rows = build_batch_customers(source_df, required_resolved, optional_resolved)

    if invalid_rows:
        render_alert(
            "warning",
            "Some rows contain invalid values and were excluded. "
            f"CSV row numbers: {', '.join(map(str, invalid_rows[:20]))}",
        )

    render_alert("info", f"Rows prepared for scoring: {len(customers)} of {len(source_df)}")

    if len(customers) == 0:
        render_alert("error", "No valid rows are available for scoring after validation.")
        return

    if not api_ok:
        render_alert("error", "API is unavailable. Verify backend connectivity before batch scoring.")
        return
    if not model_loaded:
        render_alert("error", "Model is not loaded on the backend. Check service startup logs.")
        return

    if st.button("Run batch prediction", use_container_width=True):
        with st.spinner("Running batch inference and generating analytics..."):
            predictions, error = api_predict_batch(customers)

        if error:
            render_alert("error", error)
            return

        if len(predictions) != len(customers):
            render_alert(
                "error",
                "Batch response size mismatch. "
                f"Expected {len(customers)} predictions but received {len(predictions)}.",
            )
            return

        working_df = source_df.copy().iloc[valid_indices].reset_index(drop=True)
        pred_df = pd.DataFrame(predictions)
        pred_df["recommended_action"] = pred_df.apply(
            lambda row: classify_recommended_action(
                str(row.get("risk_level", "low")),
                float(row.get("churn_probability", 0.0)),
            ),
            axis=1,
        )

        output_df = pd.concat([working_df, pred_df], axis=1)
        st.session_state["batch_output_df"] = output_df

    output_df = st.session_state.get("batch_output_df")
    if output_df is None or not isinstance(output_df, pd.DataFrame) or output_df.empty:
        return

    st.markdown('<h3 class="section-title">Results controls</h3>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        risk_filter = st.multiselect("Risk levels", options=RISK_ORDER, default=RISK_ORDER)
    with c2:
        churn_filter = st.selectbox("Predicted class", ["All", "Churn only", "No churn only"], index=0)
    with c3:
        min_probability = st.slider("Minimum probability", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    with c4:
        sort_field = st.selectbox("Sort by", ["churn_probability", "risk_level", "churn"], index=0)

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
        render_alert("warning", "No rows match the selected filters. Adjust filters to view analytics.")
        return

    st.markdown('<h3 class="section-title">Triage KPIs</h3>', unsafe_allow_html=True)
    total_customers = len(filtered)
    churn_count = int(filtered["churn"].sum())
    avg_probability = float(filtered["churn_probability"].mean())
    high_risk_count = int((filtered["risk_level"] == "high").sum())
    churn_rate = churn_count / total_customers

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_kpi_card("Total customers", f"{total_customers}")
    with k2:
        render_kpi_card("Predicted churn", f"{churn_count}")
    with k3:
        render_kpi_card("Predicted churn rate", f"{churn_rate:.1%}")
    with k4:
        render_kpi_card("Average probability", f"{avg_probability:.1%}")
    with k5:
        render_kpi_card("High-risk workload", f"{high_risk_count}")

    summary_tab, segment_tab, driver_tab, table_tab = st.tabs(
        ["Summary", "Segments", "Drivers", "Scored Table"]
    )

    with summary_tab:
        p1, p2 = st.columns(2)
        with p1:
            risk_counts = filtered["risk_level"].value_counts().reindex(RISK_ORDER, fill_value=0)
            risk_fig = go.Figure(
                data=[
                    go.Bar(
                        x=risk_counts.index.tolist(),
                        y=risk_counts.values.tolist(),
                        marker_color=[RISK_COLORS["low"], RISK_COLORS["medium"], RISK_COLORS["high"]],
                    )
                ]
            )
            risk_fig = apply_plotly_theme(risk_fig, height=340, title="Risk distribution")
            risk_fig.update_layout(xaxis_title="Risk level", yaxis_title="Customers")
            st.plotly_chart(risk_fig, use_container_width=True)

        with p2:
            prob_fig = go.Figure(
                data=[go.Histogram(x=filtered["churn_probability"], nbinsx=20, marker_color=PLOT_COLORS[0])]
            )
            prob_fig = apply_plotly_theme(prob_fig, height=340, title="Churn probability distribution")
            prob_fig.update_layout(xaxis_title="Probability", yaxis_title="Customer count")
            st.plotly_chart(prob_fig, use_container_width=True)

    with segment_tab:
        available_segments = [col for col in SEGMENT_COLUMNS if col in filtered.columns]
        if not available_segments:
            render_alert("info", "No standard segment columns found in uploaded data for segment analytics.")
        else:
            selected_segment = st.selectbox("Segment by", available_segments)
            segment_df = build_segment_breakdown(filtered, selected_segment)
            st.dataframe(segment_df, use_container_width=True, hide_index=True)

    with driver_tab:
        high_risk_subset = filtered[filtered["risk_level"] == "high"]
        driver_source = high_risk_subset if not high_risk_subset.empty else filtered
        drivers = aggregate_feature_drivers(driver_source)
        if drivers.empty:
            render_alert("info", "Feature contribution data was not present in results.")
        else:
            top_drivers = drivers.head(12)
            driver_fig = go.Figure(
                data=[
                    go.Bar(
                        y=list(reversed(top_drivers["feature"].tolist())),
                        x=list(reversed(top_drivers["total_contribution"].tolist())),
                        orientation="h",
                        marker_color=[
                            RISK_COLORS["high"] if v > 0 else RISK_COLORS["low"]
                            for v in reversed(top_drivers["total_contribution"].tolist())
                        ],
                    )
                ]
            )
            driver_fig = apply_plotly_theme(
                driver_fig,
                height=max(340, len(top_drivers) * 32),
                title="Aggregated top feature drivers",
            )
            driver_fig.update_layout(xaxis_title="Net contribution", yaxis_title="")
            st.plotly_chart(driver_fig, use_container_width=True)
            st.dataframe(top_drivers, use_container_width=True, hide_index=True)

    with table_tab:
        top_n = st.slider("Rows to display", min_value=10, max_value=200, value=40, step=10)
        top_customers = filtered.sort_values("churn_probability", ascending=False).head(top_n)
        st.dataframe(top_customers, use_container_width=True, hide_index=True)

    st.markdown('<h3 class="section-title">Exports</h3>', unsafe_allow_html=True)
    e1, e2 = st.columns(2)
    with e1:
        st.download_button(
            "Download filtered results",
            data=filtered.to_csv(index=False),
            file_name="filtered_churn_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with e2:
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
