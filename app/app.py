"""
Fake Job Posting Detector — demo app
Run with:  streamlit run app.py
(requires: pip install streamlit scikit-learn joblib)

Expects fraud_model.joblib and tfidf_vectorizer.joblib in the same folder
(produced by the project notebook).
"""

import re
import streamlit as st
import joblib

st.set_page_config(page_title="Fake Job Posting Detector", page_icon="🕵️", layout="centered")


@st.cache_resource
def load_model():
    model = joblib.load("fraud_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    return model, vectorizer


def clean_text(text: str) -> str:
    """Same cleaning function used during training — must match exactly."""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"#url_\w+#|#email_\w+#|#phone_\w+#", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict(model, vectorizer, title, company_profile, description, requirements, benefits):
    raw = " ".join([title, company_profile, description, requirements, benefits])
    cleaned = clean_text(raw)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][1]
    label = "FRAUDULENT" if pred == 1 else "LEGITIMATE"
    return label, float(prob)


st.title("🕵️ Fake Job Posting Detector")
st.caption("Paste a job posting below to check whether it looks legitimate or fraudulent. "
           "Trained on the EMSCAD dataset (17,880 postings) with TF-IDF + Logistic Regression.")

try:
    model, vectorizer = load_model()
except FileNotFoundError:
    st.error("Model files not found. Make sure `fraud_model.joblib` and `tfidf_vectorizer.joblib` "
             "are in the same folder as this app.")
    st.stop()

with st.form("posting_form"):
    title = st.text_input("Job Title", placeholder="e.g. Senior Data Analyst")
    company_profile = st.text_area("Company Profile", placeholder="Brief description of the company...", height=80)
    description = st.text_area("Job Description", placeholder="What the role involves...", height=120)
    requirements = st.text_area("Requirements", placeholder="Skills, experience, education needed...", height=100)
    benefits = st.text_area("Benefits", placeholder="Salary, perks, benefits offered...", height=80)
    submitted = st.form_submit_button("Check this posting", type="primary")

if submitted:
    if not any([title, company_profile, description, requirements, benefits]):
        st.warning("Please fill in at least one field.")
    else:
        label, prob = predict(model, vectorizer, title, company_profile, description, requirements, benefits)
        st.divider()
        if label == "FRAUDULENT":
            st.error(f"### ⚠️ {label}")
        else:
            st.success(f"### ✅ {label}")
        st.metric("Fraud probability", f"{prob:.1%}")
        st.progress(min(max(prob, 0.0), 1.0))
        st.caption(
            "This is a machine-learning estimate based on text patterns, not a verified fact. "
            "Always independently verify a company before sharing personal information or paying any fee."
        )

st.divider()
with st.expander("Try an example"):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load a legitimate example"):
            st.session_state.update({
                "example": ("Senior Data Analyst",
                             "Acme Corp is a 200-person analytics consultancy founded in 2010, based in Chicago.",
                             "We are looking for a Senior Data Analyst to join our growing team, working with SQL and Python.",
                             "3+ years experience with SQL, Python, and data visualization tools. Bachelor's degree required.",
                             "Health insurance, 401k matching, flexible PTO.")
            })
    with col2:
        if st.button("Load a scam-like example"):
            st.session_state.update({
                "example": ("Easy Data Entry - Work From Home - Earn $500/day!!!",
                             "",
                             "No experience needed! Just fill out simple forms from home and earn cash daily. Immediate start.",
                             "No qualifications needed. Must have a bank account to receive payments.",
                             "Unlimited earning potential! Be your own boss!")
            })
    st.caption("Click a button above, then re-fill the form fields with the shown text and submit.")
