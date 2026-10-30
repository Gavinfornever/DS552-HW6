"""Reusable data preparation, cross-validation, and plotting for HW6."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

SEED = 552
FEATURE_COUNT = 170


def load_aps(root=Path("data")):
    """Load the two original files; the 20-line preamble is not a CSV header."""
    train = pd.read_csv(root / "aps_failure_training_set.csv", skiprows=20, na_values="na")
    test = pd.read_csv(root / "aps_failure_test_set.csv", skiprows=20, na_values="na")
    assert train.shape == (60000, 171) and test.shape == (16000, 171)
    assert list(train.columns) == list(test.columns)
    features = [column for column in train.columns if column != "class"]
    assert len(features) == FEATURE_COUNT
    X_train = train[features].astype("float32")
    X_test = test[features].astype("float32")
    y_train = train["class"].map({"neg": 0, "pos": 1}).to_numpy()
    y_test = test["class"].map({"neg": 0, "pos": 1}).to_numpy()
    assert not np.isnan(y_train).any() and not np.isnan(y_test).any()
    return X_train, X_test, y_train, y_test


def impute_once(X_train, X_test):
    """Fit medians on training rows only, then apply them to both sets."""
    imputer = SimpleImputer(strategy="median")
    train_values = imputer.fit_transform(X_train)
    test_values = imputer.transform(X_test)
    return train_values, test_values, imputer


def coefficient_of_variation(train_values, feature_names):
    means = np.mean(train_values, axis=0)
    stds = np.std(train_values, axis=0, ddof=1)
    values = np.divide(stds, means, out=np.full_like(stds, np.nan), where=means != 0)
    return pd.DataFrame({"mean": means, "sample_sd": stds, "CV": values}, index=feature_names)


def compact_cv_table(cv_series, columns_per_row=5):
    items = list(cv_series.sort_index().items())
    rows = []
    for offset in range(0, len(items), columns_per_row):
        row = {}
        for number, (feature, value) in enumerate(items[offset:offset + columns_per_row], start=1):
            row[f"feature {number}"] = feature
            row[f"CV {number}"] = round(value, 3) if np.isfinite(value) else "undefined"
        rows.append(row)
    return pd.DataFrame(rows).fillna("")


def plot_correlation(train_values, feature_names):
    correlation = pd.DataFrame(train_values, columns=feature_names).corr()
    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    sns.heatmap(correlation, vmin=-1, vmax=1, center=0, cmap="coolwarm",
                xticklabels=False, yticklabels=False, cbar_kws={"shrink": 0.75}, ax=ax)
    ax.set_title("Training feature correlation matrix (170 features)")
    plt.show()


def plot_top_features(train_values, y_train, feature_names, top_features):
    """Plot all positives and a reproducible negative subsample for legibility."""
    rng = np.random.default_rng(SEED)
    positive = np.flatnonzero(y_train == 1)
    negative = rng.choice(np.flatnonzero(y_train == 0), 1500, replace=False)
    shown = np.r_[positive, negative]
    indexes = [feature_names.index(feature) for feature in top_features]
    fig, axes = plt.subplots(4, 4, figsize=(11.4, 8.1), constrained_layout=True)
    for ax, name, column in zip(axes.flat, top_features, indexes):
        values = np.log1p(np.maximum(train_values[shown, column], 0))
        jitter = rng.normal(0, 0.035, len(shown))
        for cls, color, label in [(0, "#377eb8", "negative"), (1, "#e41a1c", "positive")]:
            mask = y_train[shown] == cls
            ax.scatter(values[mask], cls + jitter[mask], s=5, alpha=0.28,
                       c=color, label=label)
        ax.set(xlabel=f"log(1 + {name})", ylabel="class")
        ax.set_yticks([0, 1])
    for ax in list(axes.flat)[len(top_features):]:
        ax.axis("off")
    fig.suptitle("Top-CV features: all positives and 1,500 sampled negatives")
    plt.show()

    fig, axes = plt.subplots(4, 4, figsize=(11.4, 7.6), constrained_layout=True)
    for ax, name, column in zip(axes.flat, top_features, indexes):
        transformed = np.log1p(np.maximum(train_values[:, column], 0))
        sns.boxplot(x=y_train, y=transformed, ax=ax, color="#80b1d3",
                    showfliers=False, linewidth=0.8)
        ax.set(xlabel="class (0=neg, 1=pos)", ylabel=f"log(1 + {name})")
    for ax in list(axes.flat)[len(top_features):]:
        ax.axis("off")
    fig.suptitle("Top-CV feature distributions by class (all training rows)")
    plt.show()


def forest_pipeline(balanced=False):
    forest = RandomForestClassifier(
        n_estimators=120, max_features="sqrt", min_samples_leaf=2,
        class_weight="balanced_subsample" if balanced else None,
        bootstrap=True, oob_score=True, n_jobs=6, random_state=SEED,
    )
    return ImbPipeline([("impute", SimpleImputer(strategy="median")), ("model", forest)])


def xgb_pipeline(alpha, smote=False):
    steps = [("impute", SimpleImputer(strategy="median"))]
    if smote:
        steps.extend([("scale", StandardScaler()),
                      ("smote", SMOTE(random_state=SEED, k_neighbors=5))])
    steps.append(("model", XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.85, colsample_bytree=0.85,
        objective="binary:logistic", eval_metric="logloss",
        reg_alpha=alpha, n_jobs=6, tree_method="hist", random_state=SEED,
    )))
    return ImbPipeline(steps)


def cv_xgb_alpha(X_train, y_train, alphas, smote=False):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for alpha in alphas:
        scores = cross_validate(
            xgb_pipeline(alpha, smote=smote), X_train, y_train, cv=cv,
            scoring={"accuracy": "accuracy", "balanced": "balanced_accuracy",
                     "auc": "roc_auc"}, n_jobs=1, error_score="raise",
        )
        rows.append({"reg_alpha": alpha,
                     "CV_error": 1 - scores["test_accuracy"].mean(),
                     "CV_balanced_accuracy": scores["test_balanced"].mean(),
                     "CV_AUC": scores["test_auc"].mean()})
    table = pd.DataFrame(rows)
    winner = table.sort_values(["CV_balanced_accuracy", "CV_AUC", "reg_alpha"],
                               ascending=[False, False, True]).iloc[0]
    return table, float(winner["reg_alpha"])


def evaluation_row(name, split, model, X, y):
    predicted = model.predict(X)
    scores = model.predict_proba(X)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    row = {"model": name, "split": split, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
           "error": 1 - accuracy_score(y, predicted),
           "balanced_accuracy": balanced_accuracy_score(y, predicted),
           "AUC": roc_auc_score(y, scores)}
    return row, predicted, scores


def plot_diagnostics(name, y_train, y_test, model, X_train, X_test):
    train_row, train_pred, train_scores = evaluation_row(name, "train", model, X_train, y_train)
    test_row, test_pred, test_scores = evaluation_row(name, "test", model, X_test, y_test)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), constrained_layout=True)
    for ax, y, predicted, split in [
        (axes[0], y_train, train_pred, "Train"),
        (axes[1], y_test, test_pred, "Test"),
    ]:
        ConfusionMatrixDisplay.from_predictions(
            y, predicted, labels=[0, 1], display_labels=["neg", "pos"],
            cmap="Blues", colorbar=False, ax=ax,
        )
        ax.set_title(f"{split} confusion matrix")
    for y, scores, split in [(y_train, train_scores, "train"),
                             (y_test, test_scores, "test")]:
        fpr, tpr, _ = roc_curve(y, scores)
        axes[2].plot(fpr, tpr, label=f"{split}: {roc_auc_score(y, scores):.3f}")
    axes[2].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[2].set(xlabel="False positive rate", ylabel="True positive rate",
                title="ROC (AUC in legend)")
    axes[2].legend(fontsize=8)
    fig.suptitle(name)
    plt.show()
    return [train_row, test_row]


def plot_svm_exercise():
    points = np.array([[3, 4], [2, 2], [4, 4], [1, 4],
                       [2, 1], [4, 3], [4, 1]], dtype=float)
    labels = np.array([1, 1, 1, 1, -1, -1, -1])
    model = SVC(kernel="linear", C=1e6).fit(points, labels)
    assert set(model.support_ + 1) == {2, 3, 5, 6}
    xx = np.linspace(0.4, 4.5, 150)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for label, color, name in [(1, "#d62728", "red"), (-1, "#1f77b4", "blue")]:
        mask = labels == label
        ax.scatter(points[mask, 0], points[mask, 1], s=65, color=color, label=name)
    for i, (x1, x2) in enumerate(points, 1):
        ax.annotate(str(i), (x1 + 0.06, x2 + 0.08), fontsize=9)
    ax.plot(xx, xx - 0.5, "k-", label="maximal margin")
    ax.plot(xx, xx - 0.7, "k--", alpha=0.7, label="other separator")
    ax.plot(xx, xx, ":", color="gray", lw=0.8)
    ax.plot(xx, xx - 1, ":", color="gray", lw=0.8)
    ax.scatter([3], [1.5], s=90, marker="x", color="#d62728",
               label="added red point")
    ax.set(xlim=(0.5, 4.6), ylim=(0.5, 4.7), xlabel="X1", ylabel="X2",
           title="ISLR 9.7.3: margins and a point preventing separation")
    ax.legend(fontsize=7, loc="upper left")
    plt.show()
    return model
