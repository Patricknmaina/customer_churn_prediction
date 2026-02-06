# tests/modeling/test_evaluate.py

"""
Tests for scripts/modeling/evaluate.py - unit tests for model evaluation functions.
"""

import matplotlib
import matplotlib.pyplot as plt

from scripts.modeling.evaluate import compute_classification_metrics, plot_confusion_matrix

# Use non-interactive backend for tests
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# compute_classification_metrics
# ---------------------------------------------------------------------------

class TestComputeClassificationMetrics:

    def test_perfect_predictions(self):
        metrics = compute_classification_metrics([0, 0, 1, 1], [0, 0, 1, 1])
        assert metrics["accuracy"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["f1"] == 1.0
        assert "roc_auc" not in metrics

    def test_worst_case_predictions(self):
        metrics = compute_classification_metrics([0, 0, 1, 1], [1, 1, 0, 0])
        assert metrics["accuracy"] == 0.0
        assert metrics["recall"] == 0.0

    def test_with_probabilities(self):
        metrics = compute_classification_metrics(
            [0, 0, 1, 1], [0, 0, 1, 1], y_proba=[0.1, 0.2, 0.8, 0.9]
        )
        assert "roc_auc" in metrics
        assert metrics["roc_auc"] == 1.0

    def test_without_probabilities(self):
        metrics = compute_classification_metrics([0, 1], [0, 1])
        assert "roc_auc" not in metrics

    def test_returns_dict(self):
        metrics = compute_classification_metrics([0, 1], [0, 1])
        assert isinstance(metrics, dict)

    def test_metric_values_between_zero_and_one(self):
        metrics = compute_classification_metrics(
            [0, 0, 1, 1, 0, 1], [0, 1, 1, 0, 0, 1], y_proba=[0.2, 0.6, 0.8, 0.3, 0.1, 0.7]
        )
        for value in metrics.values():
            assert 0.0 <= value <= 1.0


# ---------------------------------------------------------------------------
# plot_confusion_matrix (smoke tests)
# ---------------------------------------------------------------------------

class TestPlotConfusionMatrix:

    def test_smoke(self):
        plot_confusion_matrix([0, 0, 1, 1], [0, 1, 1, 0])
        plt.close("all")

    def test_smoke_with_labels(self):
        plot_confusion_matrix(
            [0, 0, 1, 1], [0, 1, 1, 0], class_labels=["No Churn", "Churn"]
        )
        plt.close("all")
