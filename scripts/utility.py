# scripts/utility.py

"""
Backwards-compatibility shim.

Re-exports all functions from the refactored submodules so that
existing notebooks using `from utility import ...` continue to work.

"""

# Data preparation
from scripts.data_prep.cleaning import (
    clean_nulls_and_duplicates,
    drop_highly_correlated_features,
    remove_outliers_zscore,
)

# EDA visualizations
from scripts.eda.plots import (
    categorical_churn,
    categorical_distributions,
    correlation_heatmap,
    kde_plots_with_churn,
    numerical_distribution,
)

# Modeling
from scripts.modeling.evaluate import plot_confusion_matrix
from scripts.modeling.hyperparameter_tuning import param_random_search
