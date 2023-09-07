import logging
from itertools import product
from typing import Optional

import pandas as pd
import tqdm
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA as sklearnPCA
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

N_JOBS = 20

estimators = {
    LogisticRegression(): {
        "C": [0.1, 1, 10],
        "penalty": ["l2"],
    },
    DecisionTreeClassifier(): {
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [None, 5, 10, 15],
    },
    # SVC(): {
    #     "kernel": ["linear", "poly", "rbf", "sigmoid"],
    #     "gamma": [0.1, 1, 10, 100],
    #     "C": [0.1, 1, 10, 100, 1000],
    # },
    AdaBoostClassifier(): {
        "estimator": [
            DecisionTreeClassifier(),
            RandomForestClassifier(),
        ],
        "n_estimators": [10, 25, 50],
    },
    RandomForestClassifier(): {
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [None, 5, 10, 15],
    },
    BaggingClassifier(): {
        "estimator": [
            DecisionTreeClassifier(),
            RandomForestClassifier(),
        ],
        "n_estimators": [10, 25, 50],
    },
    GradientBoostingClassifier(): {
        "n_estimators": [10, 25, 50],
        "max_depth": [1, 2, 5, 10],
    },
    KNeighborsClassifier(): {"n_neighbors": [5, 20, 50, 100]},
    RadiusNeighborsClassifier(): {"radius": [1, 20, 50, 100, 300]},
}

dim_reductors = [
    sklearnPCA(20),
    # sklearnPCA(0.90),
    # sklearnPCA(0.95),
    sklearnPCA(10),
    sklearnPCA(5),
    # TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300),  # use after reducing features count < 50
    KernelPCA(n_components=5, kernel="rbf", n_jobs=N_JOBS),
    KernelPCA(n_components=10, kernel="rbf", n_jobs=N_JOBS),
    KernelPCA(n_components=5, kernel="poly", n_jobs=N_JOBS),
    KernelPCA(n_components=10, kernel="poly", n_jobs=N_JOBS),
    LocallyLinearEmbedding(n_components=10, n_neighbors=40, n_jobs=N_JOBS),
    LocallyLinearEmbedding(n_components=5, n_neighbors=40, n_jobs=N_JOBS),
    Isomap(n_components=5, n_neighbors=40, n_jobs=N_JOBS),
]

samplers: list[Optional[BaseOverSampler]] = [
    None,
    # SMOTE(sampling_strategy=0.5),
    # SMOTE(sampling_strategy=0.7),
    # SMOTE(),
    # ADASYN(sampling_strategy=0.5),
    # ADASYN(sampling_strategy=0.7),
    RandomUnderSampler(),
    ADASYN(),
]

df = pd.read_csv("./dataset.csv")

X = df.drop(["target"], axis=1)
y = df["target"]

X = StandardScaler().fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)


def score(y_true, y_pred):
    return roc_auc_score(y_true, y_score=y_pred, average="weighted")


roc_auc_scorer = make_scorer(roc_auc_score, average="weighted")
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

    run_result = (
        over_sampler,
        dim_reductor,
        estimator,
        grid.best_params_,
        grid.best_score_,
        accuracy_score(y_test, y_pred),
        score(y_test, y_pred),
    )

    run_results.append(run_result)
    logging.debug(run_result)

pd.DataFrame(run_results).to_csv("run_results.csv")
