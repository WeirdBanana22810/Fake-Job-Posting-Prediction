"""
End-to-end training pipeline for the Fake Job Posting Detection project.

Usage:
    python -m src.train_pipeline --data data/fake_job_postings.csv

Trains all four models, evaluates them, tunes Logistic Regression, and
saves the tuned model + TF-IDF vectorizer to app/ for use by the demo app.
"""

import argparse
import warnings

import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression

from src.preprocessing import load_data, build_combined_text
from src.features import build_tfidf
from src.modeling import get_models, train_models, evaluate_all, RANDOM_STATE

warnings.filterwarnings("ignore")


def main(data_path: str, model_out: str, vectorizer_out: str):
    print(f"Loading data from {data_path} ...")
    df = load_data(data_path)
    df = build_combined_text(df)
    print(f"Loaded {len(df)} postings ({df['fraudulent'].mean()*100:.2f}% fraudulent).")

    print("Building TF-IDF features ...")
    tfidf, X_tfidf = build_tfidf(df["full_text_clean"])
    y = df["fraudulent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print("Training models ...")
    trained_models = train_models(get_models(), X_train, y_train)

    print("Evaluating models ...")
    results_df = evaluate_all(trained_models, X_test, y_test)
    print(results_df)

    print("Tuning Logistic Regression (GridSearchCV) ...")
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        {"C": [0.1, 1, 5, 10]}, scoring="f1", cv=3, n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    print(f"Best params: {grid.best_params_}  (CV F1={grid.best_score_:.4f})")

    joblib.dump(best_model, model_out)
    joblib.dump(tfidf, vectorizer_out)
    print(f"Saved model to {model_out} and vectorizer to {vectorizer_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the fake job posting detector.")
    parser.add_argument("--data", default="data/fake_job_postings.csv", help="Path to the EMSCAD CSV file.")
    parser.add_argument("--model-out", default="app/fraud_model.joblib", help="Where to save the trained model.")
    parser.add_argument("--vectorizer-out", default="app/tfidf_vectorizer.joblib", help="Where to save the fitted vectorizer.")
    args = parser.parse_args()
    main(args.data, args.model_out, args.vectorizer_out)
