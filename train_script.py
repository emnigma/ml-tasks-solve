import pandas as pd

RS = 42

df = pd.read_csv("dataset.csv")

from imblearn.over_sampling import ADASYN
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop(["target"], axis=1)
y = df["target"]

scaler = StandardScaler()

X = scaler.fit_transform(X, y)

# split first, then resample
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RS
)

over_sampler = ADASYN(random_state=RS)
reductor = KernelPCA(kernel="poly", n_components=5, random_state=RS)

from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    RandomForestClassifier,
)
from sklearn.neighbors import KNeighborsClassifier

bagging_clf = BaggingClassifier(
    estimator=RandomForestClassifier(random_state=RS), n_estimators=50, random_state=RS
)
rf_clf = RandomForestClassifier(criterion="entropy", max_depth=None, random_state=RS)
knn_clf = KNeighborsClassifier(n_neighbors=5)
adaboost_clf = AdaBoostClassifier(
    estimator=RandomForestClassifier(), n_estimators=25, random_state=RS
)

from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import cross_val_score

X_train_resampled, y_train_resampled = over_sampler.fit_resample(X_train, y_train)

X_reducted = reductor.fit_transform(X_train_resampled)

for est in [bagging_clf, rf_clf, knn_clf, adaboost_clf]:
    print(est)
    est.fit(X_reducted, y_train_resampled)

    print(
        cross_val_score(
            est,
            X_reducted,
            y_train_resampled,
            scoring=make_scorer(roc_auc_score, average="weighted"),
            cv=5,
        )
    )

    X_pred = reductor.transform(X_test)
    y_pred = est.predict(X_pred)

    print(roc_auc_score(y_true=y_test, y_score=y_pred, average="weighted"))


test_df = pd.read_csv("test.csv", index_col="id")

final_X_test_scaled = scaler.transform(test_df)
final_X_test_reducted = reductor.transform(final_X_test_scaled)
final_y_test_pred = knn_clf.predict(final_X_test_reducted)

submission = pd.DataFrame(final_y_test_pred, columns=["target"])
submission.to_csv("submission.csv", index_label="id")

print(submission.value_counts())
