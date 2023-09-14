import logging
from itertools import product
from typing import Optional

import pandas as pd
import tqdm
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(
    level=logging.DEBUG,
    filename="learn.log",
    filemode="a",
    format="%(asctime)s - p%(process)d: %(name)s - [%(levelname)s]: %(message)s",
)

N_JOBS = 8
RS=42

estimators = {
    # LogisticRegression(): {
    #     "C": [0.1, 1, 10],
    #     "penalty": ["l2"],
    # },
    DecisionTreeClassifier(random_state=RS): {
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [None, 5, 10, 15],
    },
    # SVC(): {
    #     "kernel": ["linear", "poly", "rbf", "sigmoid"],
    #     "gamma": [0.1, 1, 10, 100],
    #     "C": [0.1, 1, 10, 100, 1000],
    # },
    AdaBoostClassifier(random_state=RS): {
        "estimator": [
            DecisionTreeClassifier(random_state=RS),
            RandomForestClassifier(random_state=RS),
        ],
        "n_estimators": [10, 25, 50],
    },
    RandomForestClassifier(random_state=RS): {
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [None, 5, 10, 15],
    },
    BaggingClassifier(random_state=RS): {
        "estimator": [
            DecisionTreeClassifier(random_state=RS),
            RandomForestClassifier(random_state=RS),
        ],
        "n_estimators": [10, 25, 50],
    },
    GradientBoostingClassifier(random_state=RS): {
        "n_estimators": [10, 25, 50],
        "max_depth": [1, 2, 5, 10],
    },
    KNeighborsClassifier(): {"n_neighbors": [5, 20, 50, 100]},
    # RadiusNeighborsClassifier(): {"radius": [1, 20, 50, 100, 300]},
}

dim_reductors = [
    PCA(20, random_state=RS),
    # PCA(0.90),
    # PCA(0.95),
    PCA(10, random_state=RS),
    PCA(5, random_state=RS),
    # TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300),  # use after reducing features count < 50
    KernelPCA(n_components=5, kernel="rbf", n_jobs=N_JOBS, random_state=RS),
    KernelPCA(n_components=10, kernel="rbf", n_jobs=N_JOBS, random_state=RS),
    KernelPCA(n_components=5, kernel="poly", n_jobs=N_JOBS, random_state=RS),
    KernelPCA(n_components=10, kernel="poly", n_jobs=N_JOBS, random_state=RS),
    LocallyLinearEmbedding(n_components=10, n_neighbors=40, n_jobs=N_JOBS, random_state=RS),
    LocallyLinearEmbedding(n_components=5, n_neighbors=40, n_jobs=N_JOBS, random_state=RS),
    # Isomap(n_components=5, n_neighbors=5, n_jobs=N_JOBS, random_state=RS),
]

samplers: list[Optional[BaseOverSampler]] = [
    None,
    SMOTE(sampling_strategy=0.5, random_state=RS),
    # SMOTE(sampling_strategy=0.7),
    # SMOTE(),
    ADASYN(sampling_strategy=0.5, random_state=RS),
    # ADASYN(sampling_strategy=0.7),
    RandomUnderSampler(random_state=RS),
    ADASYN(random_state=RS),
]

df = pd.read_csv("./dataset.csv")

X = df.drop(["target"], axis=1)
y = df["target"]

X = StandardScaler().fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def score(y_true, y_pred):
    return roc_auc_score(y_true, y_score=y_pred, average="weighted")


roc_auc_scorer = make_scorer(roc_auc_score, average="weighted", needs_proba=True)
run_results = []


data = list(product(samplers, dim_reductors, list(estimators.keys())))

for over_sampler, dim_reductor, estimator in tqdm.tqdm(data):
    # logging.debug(f"start: {over_sampler=}, {dim_reductor=}, {estimator=}")
    if over_sampler:
        X_train_resampled, y_train_resampled = over_sampler.fit_resample(
            X_train, y_train
        )
    else:
        X_train_resampled, y_train_resampled = X_train, y_train

    X_reducted = dim_reductor.fit_transform(X_train_resampled)

    param_grid = estimators[estimator]
    grid = GridSearchCV(
        estimator, param_grid, scoring=roc_auc_scorer, cv=5, n_jobs=N_JOBS
    )
    grid.fit(X_reducted, y_train_resampled)

    X_pred = dim_reductor.transform(X_test)
    y_pred = grid.predict(X_pred)
    y_pred_probs = grid.predict_proba(X_pred)[:, 1]

    run_result = (
        over_sampler,
        dim_reductor,
        estimator,
        grid.best_params_,
        grid.best_score_,
        accuracy_score(y_test, y_pred),
        score(y_test, y_pred_probs),
    )

    run_results.append(run_result)
    logging.debug(run_result)

pd.DataFrame(run_results).to_csv("run_results.csv")
