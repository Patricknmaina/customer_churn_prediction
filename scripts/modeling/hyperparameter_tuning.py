# scripts/modeling/hyperparameter_tuning.py

"""
Function for randomized hyperparameter search with cross-validation.
"""

from sklearn.model_selection import RandomizedSearchCV


def param_random_search(model, param_dist, X, y, n_iter=10, cv=5, verbose=True, n_jobs=-1):
    """
    Performs randomized hyperparameter search with cross-validation.

    Args:
        model : estimator object
            The classification model to tune.
        param_dist : dict
            Dictionary with parameters names (`str`) as keys and distributions or lists of parameters to try.
        X : pd.DataFrame or np.ndarray
            Training features.
        y : pd.Series or np.ndarray
            Training labels.
        n_iter : int, default=10
            Number of parameter settings sampled.
        cv : int, default=5
            Number of cross-validation folds.
        verbose : bool or int
            Controls the verbosity of the output.
        n_jobs : int, default=-1
            Number of jobs to run in parallel (-1 means use all processors).

    Returns:
        RandomizedSearchCV object
    """

    # Instantiate the RandomizedSearchCV function
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='recall',
        verbose=verbose,
        n_jobs=n_jobs # set to -1 for utilization of all CPU cores (parallelization)
    )

    # Fit the random search with train data
    random_search.fit(X, y)

    # Determine the best parameters, score and estimators
    print("Best model hyperparameters:", random_search.best_params_)
    print("Best model accuracy:", random_search.best_score_)
    print("Best model estimators:", random_search.best_estimator_)

    return random_search
