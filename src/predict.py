"""
Final prediction system: classify a new, unseen job posting as
fraudulent or legitimate using a trained model + vectorizer.
"""

import joblib
from .preprocessing import clean_text


def load_artifacts(model_path: str, vectorizer_path: str):
    """Load a previously trained model and its matching vectorizer."""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def predict_job_posting(model, vectorizer, title="", company_profile="",
                         description="", requirements="", benefits=""):
    """Classify a new job posting as fraudulent or legitimate.

    Returns a dict: {"prediction": "FRAUDULENT" | "LEGITIMATE", "fraud_probability": float}
    """
    raw = " ".join([title, company_profile, description, requirements, benefits])
    cleaned = clean_text(raw)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][1]
    label = "FRAUDULENT" if pred == 1 else "LEGITIMATE"
    return {"prediction": label, "fraud_probability": round(float(prob), 4)}
