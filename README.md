# **Customer Churn Prediction**

Author: [Patrick Maina](https://github.com/Patricknmaina)

A production ML system that predicts customer churn for a Telecommunication company. The project includes a trained XGBoost model, a REST API backend deployed on Railway, and an interactive Streamlit dashboard deployed on Streamlit Community Cloud.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        GitHub                           │
│  Push to main → CI/CD (GitHub Actions)                  │
│    └── Run tests                                        │
│    └── Report status to Railway                         │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────┐
        │      Railway             │
        │      FastAPI API         │  ◄── Streamlit Community Cloud
        │  /predict  /predict/batch│       (streamlit/app.py)
        │  /health                 │
        └──────────────────────────┘
```

- **Backend**: FastAPI on Railway — handles all predictions, preprocessing, and SHAP explanations (Explainable AI)
- **Frontend**: Streamlit on Streamlit Community Cloud — pure HTTP client, calls the backend API
- **Registry**: Railway builds and deploys from this repository's Dockerfile
- **CI/CD**: GitHub Actions runs tests, and Railway deploys after checks pass

## Features

- **Single Prediction** — Enter customer details and get an instant churn prediction with probability score, risk level (low/medium/high), and a SHAP feature contribution chart explaining the top 10 drivers
- **Batch Prediction** — Upload a CSV of customers, get predictions for all of them, and download the results
- **Explainability** — SHAP values aggregated back to original feature names, showing which factors increase or decrease churn risk
- **REST API** — `POST /predict` and `POST /predict/batch` endpoints for programmatic access

---

## Business Understanding

### Introduction
The telecommunications industry has become very competitive over the years, with customer retention emerging as a critical challenge. One of the major issues facing telecom providers is customer churn; a scenario where users discontinue their service, either due to dissatisfaction from the provider, or due to the availability of better alternatives. High churn rates can significantly impact a company's overall revenue and scaling potential.

### Problem Statement
SyriaTel, a leading telecom provider, is experiencing a significant loss of customers who are choosing to leave its services for competitors. To address this challenge, the company seeks to build a robust predictive model capable of identifying customers who are at risk of churning. By leveraging data-driven insights and predictive modeling, SyriaTel aims to understand the key drivers of customer attrition, determine methods of improving long-term retention, and enhance long-term customer loyalty and profitability.

### Objectives
- Determine the key characteristics and behavior patterns that likely contribute to customer churn
- Build a robust predictive model that identifies customers with a high likelihood of discontinuing their service
- Provide data-driven insights and recommendations that will proactively engage and retain high-risk customers

## Dataset

This project uses the [Telecom Churn Dataset](https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset) from Kaggle (3,333 customers). Key attributes include:

- **State and Area Code** — geographic identifiers
- **International and Voice Mail Plans** — subscription plan flags
- **Call rates** — day, night, evening, and international charge rates
- **Customer Service Calls** — number of support interactions

## Methodology

### 1. Data Exploration
Loading, inspecting for missing values/duplicates, and computing descriptive statistics.

### 2. Data Cleaning
Standardizing column names, converting `Area_Code` to object type, dropping `Phone_Number`, and removing highly correlated features (threshold: 0.9).

### 3. Exploratory Data Analysis
Univariate and bivariate analysis using distribution plots, correlation heatmaps, box plots, and bar charts.

### 4. Data Preprocessing
One-hot encoding for categorical features, MinMaxScaler for numeric features, and SMOTENC for class imbalance.

### 5. Predictive Modeling
Trained six models (Logistic Regression as baseline, Decision Tree, Random Forest, Gradient Boosting, XGBoost, and others) evaluated on recall and AUC-ROC.

### 6. Model Evaluation & Hyperparameter Tuning
ROC curves, AUC scores, confusion matrices, and RandomizedSearchCV for the top two models.

---

## 📈 Crucial Visualizations

**Analyzing Customer Service Calls by Rate of Churn**

![customer_service_churn](images/customer_service_churn.jpg)

**Distribution of Numerical Features**

![numerical_dist](images/numerical_distribution.jpg)

**Feature Correlation Heatmap**

![feature_heatmap](images/feature_heatmap.png)

---

## Conclusion

The XGBoost Classifier achieved a recall score of `0.82` and AUC of `0.911`. The Gradient Boosting model achieved a slightly higher AUC of `0.921` but lower recall of `0.81`. Given the nature of churn prediction where missing a churner is more costly than a false alarm, the **XGBoost model** is recommended. Key churn drivers identified: `Customer_Service_Calls`, `Total_Day_Charge`, and `International_Plan`.

## Business Recommendations

1. **Targeted Incentives for High-Churn Area Codes** — Customers in area codes `415` and `510` exhibit higher churn tendencies. Offering specialized discounts, loyalty rewards, or exclusive promotions in these regions can serve as an effective retention incentive.

2. **Enhance Customer Service Efficiency** — A high number of customer service interactions is correlated with increased churn. Investing in staff training and better conflict resolution frameworks can significantly boost customer satisfaction.

3. **State-Specific Retention Strategies** — States such as Texas, New Jersey, Maryland, Miami, and New York report above-average churn. Developing localized marketing and enhanced customer support in these regions would strengthen retention.

4. **Review and Optimize Call Rate Plans** — Customers who churn often experience high day, evening, night, and international call rates. Introducing more competitive or bundled plans could make services more attractive.

---

## Local Development

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)

### Setup

```bash
git clone https://github.com/Patricknmaina/customer_churn_prediction.git
cd customer_churn_prediction
uv sync
```

### Run locally with Docker Compose

```bash
docker compose up
```

This starts both services:
- FastAPI backend at `http://localhost:8000`
- Streamlit dashboard at `http://localhost:8501`

### Run tests

```bash
uv run pytest tests/ -v --tb=short
```

### Retrain the model

```bash
uv run train
```

This runs the full training pipeline and saves updated artifacts to `artifacts/`.

## Deployment

### Backend — DigitalOcean App Platform (Legacy)

The backend is deployed as a Docker container via DigitalOcean App Platform. The CI/CD pipeline handles all deployments automatically on push to `main`.

#### One-time bootstrap (first deploy)

**1. Authenticate with DigitalOcean:**
```bash
doctl auth init
```

**2. Create the container registry:**
```bash
doctl registry create customer-churn-registry --subscription-tier basic
```

**3. Build and push the image manually:**
```bash
doctl registry login --expiry-seconds 1200
docker build --target api -t registry.digitalocean.com/customer-churn-registry/churn-api:latest .
docker push registry.digitalocean.com/customer-churn-registry/churn-api:latest
```

**4. Create the App Platform app:**
```bash
doctl apps create --spec .do/app.yaml
```

**5. Get the App ID and live URL:**
```bash
doctl apps list
doctl apps get <APP_ID> --format DefaultIngress
```

#### CI/CD (automatic after bootstrap)

Every push to `main` automatically:
1. Runs tests (`uv run pytest`)
2. Builds a new Docker image tagged with the commit SHA and `latest`
3. Pushes the image to DOCR
4. Triggers a new App Platform deployment

**Required GitHub repository secrets:**

| Secret | Value |
|--------|-------|
| `DIGITALOCEAN_ACCESS_TOKEN` | Your DigitalOcean API token |
| `DOCR_REGISTRY` | `customer-churn-registry` |
| `DIGITALOCEAN_APP_ID` | App ID from `doctl apps list` |

Add these at: **GitHub repo → Settings → Secrets and variables → Actions**

---

### Backend — Railway (Migration Path)

This repo now includes `railway.toml` for Railway config-as-code.

Current Railway configuration:
- Builder: `DOCKERFILE`
- Dockerfile path: `Dockerfile.api`
- Healthcheck path: `/health`
- Healthcheck timeout: `300` seconds
- Restart policy: `ON_FAILURE` (max retries: `10`)
- Replicas: `1`

#### One-time setup on Railway

1. Create a new Railway project and service from this GitHub repository.
2. Confirm the service is using the repo `Dockerfile` (or leave default detection).
3. In service variables, set:
   - `PYTHONUNBUFFERED=1`
4. Deploy once and verify `GET /health` returns:
   - `status: "healthy"`
   - `model_loaded: true`
5. Copy the public Railway backend URL for frontend configuration.

#### Notes

- The API container now binds to Railway's dynamic port using `${PORT:-8000}`.
- Model artifacts are baked into the image (`artifacts/`), so inference works on Railway without external storage.
- If you retrain locally, commit updated `artifacts/` and redeploy.

#### CI/CD on Railway

GitHub Actions now runs tests only. Deployment is handled by Railway GitHub Autodeploy.

1. Open your Railway service.
2. Enable GitHub Autodeploy for `main`.
3. Enable `Wait for CI` in Railway deployment settings.
4. Select this workflow as the required check: `CI — Backend Tests`.

With this setup:
1. A push to `main` triggers GitHub tests.
2. Railway deploys only after CI succeeds.
3. Failed CI blocks deployment.

---

### Frontend — Streamlit Community Cloud

**1.** Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub

**2.** Click **Create app** → **Deploy a public app from GitHub**

**3.** Fill in:
- Repository: `Patricknmaina/customer_churn_prediction`
- Branch: `main`
- Main file path: `streamlit/app.py`

**4.** Click **Advanced settings** and add:
```
API_URL = https://<your-railway-backend>.up.railway.app
```

**5.** Click **Deploy!**

Community Cloud reads `streamlit/requirements.txt`, installs the four frontend dependencies, and serves the app at `https://<your-app-name>.streamlit.app`.

---

### Retraining the Model

When new data is available or the model needs updating:

```bash
# 1. Retrain locally
uv run train

# 2. Commit the updated artifacts
git add artifacts/
git commit -m "retrain model - <reason>"
git push
```

GitHub CI validates the push and Railway deploys the updated image after checks pass.

## Technologies

| Category | Tools |
|----------|-------|
| **ML** | XGBoost 3.0, scikit-learn 1.6, SHAP 0.42+, imbalanced-learn 0.13 |
| **API** | FastAPI 0.115+, Uvicorn 0.32+, Pydantic 2.10+ |
| **Frontend** | Streamlit 1.40+, Plotly 6.1, pandas 2.2, requests |
| **Packaging** | uv, joblib |
| **Containerization** | Docker (multi-stage build), Docker Compose |
| **CI/CD** | GitHub Actions |
| **Hosting** | Railway (backend), Streamlit Community Cloud (frontend) |
| **Registry** | Railway-managed deployment pipeline |
| **Testing** | pytest 8+, pytest-cov, httpx |

## Repository Structure

```
customer_churn_prediction/
├── api/                        # FastAPI backend
│   ├── main.py                 # App entrypoint, endpoints (/health, /predict, /predict/batch)
│   └── schemas.py              # Pydantic request/response models
├── streamlit/                  # Streamlit dashboard
│   ├── app.py                  # Interactive UI (Home, Single Prediction, Batch Prediction tabs)
│   └── requirements.txt        # Frontend-only dependencies for Streamlit Community Cloud
├── scripts/                    # ML pipeline
│   ├── config.py               # Central constants (paths, features, thresholds, model params)
│   ├── data_prep/              # Cleaning and preprocessing
│   ├── eda/                    # EDA visualization utilities
│   ├── modeling/               # Training, evaluation, hyperparameter tuning
│   └── serving/                # ChurnPredictor inference engine (predict.py)
├── tests/                      # Full test suite mirroring scripts/ structure
├── artifacts/                  # Serialized model artifacts (committed for CI/CD builds)
│   ├── model.joblib
│   ├── scaler.joblib
│   ├── encoder.joblib
│   ├── label_encoder.joblib
│   ├── feature_names.json
│   └── dropped_features.json
├── .do/
│   └── app.yaml                # DigitalOcean App Platform deployment spec
├── .github/workflows/
│   └── deploy-api.yml          # CI pipeline (backend tests)
├── Dockerfile                  # Multi-stage build (api target, streamlit target)
├── docker-compose.yml          # Local development (both services)
├── pyproject.toml              # Project metadata and dependencies
└── data/
    └── churn.csv               # Raw dataset (Kaggle Telecom Churn)
```

## License
MIT License - see LICENSE file for details
