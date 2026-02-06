# tests/modeling/test_hyperparameter_tuning.py

"""
Tests for scripts/modeling/hyperparameter_tuning.py - unit tests for hyperparameter tuning functions.
"""

import numpy as np
import pytest
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

from scripts.modeling.hyperparameter_tuning import param_random_search


@pytest.fixture
def search_result():
    """Run a minimal RandomizedSearchCV and return the result."""
    model = DecisionTreeClassifier(random_state=42)
    param_dist = {"max_depth": [1, 2, 3], "min_samples_split": [2, 3]}
    X = np.random.RandomState(42).rand(50, 3)
    y = np.array([0] * 25 + [1] * 25)
    return param_random_search(
        model, param_dist, X, y, n_iter=2, cv=2, verbose=0, n_jobs=1
    )


@pytest.mark.slow
class TestParamRandomSearch:

    def test_returns_randomized_search_cv(self, search_result):
        assert isinstance(search_result, RandomizedSearchCV)

    def test_has_best_params(self, search_result):
        assert hasattr(search_result, "best_params_")
        assert "max_depth" in search_result.best_params_

    def test_has_best_score(self, search_result):
        assert hasattr(search_result, "best_score_")
        assert 0 <= search_result.best_score_ <= 1

    def test_has_best_estimator(self, search_result):
        assert hasattr(search_result, "best_estimator_")
        assert isinstance(search_result.best_estimator_, DecisionTreeClassifier)

    def test_scoring_is_recall(self, search_result):
        assert search_result.scoring == "recall"
