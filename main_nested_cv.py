"""
Parallelized experiment for quantification of feature scaling leakage
in voice pathology detection using SVD and VOICED datasets.
"""
import json
import os
import warnings

from pathlib import Path
from time import time

import pandas as pd

from itertools import product
from datetime import datetime
from sklearn import set_config
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import (MinMaxScaler, StandardScaler,
                                   MaxAbsScaler, RobustScaler, QuantileTransformer)
from sklearn.pipeline import Pipeline
from param_grid import set_param_grid


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
    set_config(
        assume_finite = True,
        working_memory=4096,
        skip_parameter_validation=True,
    )
    RANDOM_STATE = 42 # for consistency of stochastic scalers / classifiers
    EXPERIMENTS = 100 # number of repetitions for each combination
    LEAKAGE = [True, False] # scaling introducing leakage / correct scaling
    STRATIFY = [True, False] # stratified split which respects the class ratio / random split

    # Selected scaling transformations
    SCALERS = [MaxAbsScaler(), MinMaxScaler(), StandardScaler(), RobustScaler(),
               QuantileTransformer(output_distribution="normal", random_state=RANDOM_STATE)]

    # Experiments tested on two different data sources
    DATASETS = [
        dict(name="svd", data=pd.read_csv(Path("data", "flattened_features.csv")), n_features=165),
        dict(name="voiced", data=pd.read_csv(Path("data", "voiced_features_8000_fft.csv")), n_features=164)
    ]

    # File name
    file_name = f"n_cv_scaling_leakage_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results = pd.DataFrame(columns=["dataset", "scaler", "stratify", "classifier", "seed", "uar_correct", "uar_leakage"])

    # Prepare all combinations of scalers and stratification splits as well as a number of experiments to reduce the
    # number of nested loops
    combinations = product(range(EXPERIMENTS), SCALERS, STRATIFY)
    for split_seed, scaler, stratify in combinations:
        start = time()
        print(f"Starting for {scaler.__class__.__name__}, {'stratified' if stratify else 'random'} split, "
              f"seed {split_seed}...", end="")
        # Set the splitter based on the stratification setting
        if stratify:
            inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=split_seed)
            outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=split_seed)
        else:
            inner_cv = KFold(n_splits=5, shuffle=True, random_state=split_seed)
            outer_cv = KFold(n_splits=5, shuffle=True, random_state=split_seed)

        # Loop through the datasets
        for dataset in DATASETS:
            print(f"\n\t- dataset: {dataset['name']}")
            print("\t\t- done: ", end="")
            # Load data
            dataset_name = dataset["name"]
            y_orig = dataset["data"]["pathology"]
            X_orig = dataset["data"].drop(columns=["pathology", "session_id"])

            X_leakage = X_orig.copy()
            X_leakage = scaler.fit_transform(X_leakage)

            # Generate a parameter grid for each dataset as the MLP is based on the number of dataset features
            param_grid = set_param_grid(dataset["n_features"], random_state=RANDOM_STATE)

            # Loop through the parameter grid to optimize each classifier
            for clf_name, clf_settings in param_grid.items():
                # Save classifier settings for optimization
                clf = clf_settings["estimator"]
                param_space = clf_settings["param_space"]
                n_iter = 100

                # Prepare data based on whether leakage is introduced or not
                leaked_scores = None
                correct_scores = None
                for leakage in LEAKAGE:
                    # In case of leakage, first transform the whole dataset, then conduct randomized search with
                    # cross-validation to find the optimal settings
                    if leakage:
                        clf_search = RandomizedSearchCV(estimator=Pipeline([("model", clf)]),
                                                        param_distributions=param_space, cv=inner_cv,
                                                        n_iter=n_iter, n_jobs=-1, random_state=split_seed,
                                                        scoring="balanced_accuracy")
                        nested_score = cross_val_score(clf_search, X=X_leakage, y=y_orig, cv=outer_cv, scoring="balanced_accuracy")

                    # In case of correct methodology, fit the scaler on the training set which is used for
                    # the hyperparameter optimization and then transform the test set.
                    else:
                        clf_search = RandomizedSearchCV(estimator=Pipeline([("scaler", scaler), ("model", clf)]),
                                                        param_distributions=param_space, cv=inner_cv,
                                                        n_iter=n_iter, n_jobs=-1, random_state=split_seed,
                                                        scoring="balanced_accuracy")
                        nested_score = cross_val_score(clf_search, X=X_orig, y=y_orig, cv=outer_cv, scoring="balanced_accuracy")

                    if leakage:
                        leaked_scores = [nested_score.mean()]
                    else:
                        correct_scores = [nested_score.mean()]

                # Save the results for one iteration of seed-scaler-stratification-dataset-classifier combination
                result_row = ([dataset["name"], scaler.__class__.__name__, stratify, clf_name, split_seed] +
                              correct_scores + leaked_scores)
                results.loc[len(results), :] = result_row
                results.to_csv(file_name, index=False, encoding="utf8")

                print(f"{clf_name}, ", end="")

        print(f"\n...done in {time() - start:.2f}s")
        print("#"*50)
