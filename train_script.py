import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler

RS = 42

df = pd.read_csv("dataset.csv")

X = df.drop(["target"], axis=1)
y = df["target"]

scaler = StandardScaler()

X = scaler.fit_transform(X, y)

over_sampler = ADASYN(sampling_strategy=0.5, random_state=RS)

from imblearn.pipeline import make_pipeline
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

poly_kernel_pca_rf_clf = make_pipeline(
    KernelPCA(kernel="poly", n_components=5, random_state=RS),
    RandomForestClassifier(criterion="entropy", max_depth=None, random_state=RS),
)
poly_kernel_pca_adaboost_clf = make_pipeline(
    KernelPCA(kernel="poly", n_components=5, random_state=RS),
    AdaBoostClassifier(
        estimator=RandomForestClassifier(), n_estimators=10, random_state=RS
    ),
)

from statistics import mean

from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import cross_val_score

X_train_resampled, y_train_resampled = over_sampler.fit_resample(X, y)
best_cls = (None, 0)

for est in [
    poly_kernel_pca_rf_clf,
    poly_kernel_pca_adaboost_clf,
]:
    print(est)
    est.fit(X_train_resampled, y_train_resampled)

    score = mean(
        cross_val_score(
            est,
            X_train_resampled,
            y_train_resampled,
            scoring=make_scorer(roc_auc_score, average="weighted", needs_proba=True),
            cv=5,
        )
    )

    if best_cls[1] < score:
        best_cls = (est, score)

    print(score)

print("--------------------------------------------")
print("best estimator: ", best_cls[0])


test_df = pd.read_csv("test.csv", index_col="id")

final_X_test_scaled = scaler.transform(test_df)
final_y_test_pred = best_cls[0].predict(final_X_test_scaled)

submission = pd.DataFrame(final_y_test_pred, columns=["target"])
submission.to_csv("submission.csv", index_label="id")
