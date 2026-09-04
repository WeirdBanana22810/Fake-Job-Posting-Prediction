"""
Model definitions, training, and evaluation utilities.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
)

RANDOM_STATE = 42


def get_models() -> dict:
    """Return the four classifiers used in this project, ready to fit."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
    }


def train_models(models: dict, X_train, y_train) -> dict:
    """Fit every model in `models` on the training data."""
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained


def get_scores(model, X_test, y_test):
    """Compute accuracy/precision/recall/F1/ROC-AUC for a fitted model."""
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:  # LinearSVC has no predict_proba; use decision_function for ROC-AUC
        y_score = model.decision_function(X_test)
    scores = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_score),
    }
    return scores, y_pred, y_score


def evaluate_all(trained_models: dict, X_test, y_test) -> pd.DataFrame:
    """Evaluate every trained model and return a results DataFrame."""
    results = {}
    for name, model in trained_models.items():
        scores, _, _ = get_scores(model, X_test, y_test)
        results[name] = scores
    return pd.DataFrame(results).T.round(4)
