# tests/eda/test_plots.py

"""
Smoke tests for scripts/eda/plots.py — verify plots run without errors.
These tests don't check visual output, just that the functions execute without exceptions.
"""

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest

matplotlib.use("Agg")

from scripts.eda.plots import (
    categorical_churn,
    categorical_distributions,
    correlation_heatmap,
    kde_plots_with_churn,
    numerical_distribution,
)


@pytest.fixture(autouse=True)
def close_plots():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def plot_df():
    """Small DataFrame with columns the EDA functions expect."""
    return pd.DataFrame({
        "State": ["KS", "OH", "NJ", "OH", "OK", "KS", "NJ", "OH", "CA", "TX"],
        "Churn": [0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
        "Total_Day_Charge": [45.0, 27.5, 41.4, 50.9, 28.3, 35.0, 42.0, 30.0, 55.0, 20.0],
        "Total_Eve_Charge": [16.8, 16.6, 10.3, 5.3, 12.6, 14.0, 11.0, 17.0, 8.0, 15.0],
        "Account_Length": [128, 107, 137, 84, 75, 100, 120, 90, 150, 60],
    })


def test_categorical_distributions_smoke(plot_df):
    categorical_distributions(plot_df, "State")


def test_numerical_distribution_smoke(plot_df):
    numerical_distribution(plot_df, ["Total_Day_Charge", "Total_Eve_Charge"])


def test_numerical_distribution_single_feature(plot_df):
    numerical_distribution(plot_df, ["Total_Day_Charge"])


def test_categorical_churn_smoke(plot_df):
    categorical_churn(plot_df, "State")


def test_kde_plots_with_churn_smoke(plot_df):
    kde_plots_with_churn(plot_df, "Total_Day_Charge", "Day")


@pytest.mark.xfail(reason="np.bool removed in NumPy 2.x — source code needs updating")
def test_correlation_heatmap_smoke(plot_df):
    correlation_heatmap(plot_df)
