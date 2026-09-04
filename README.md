# Fake Job Posting Detection

A machine learning system that classifies job postings as **fraudulent** or
**legitimate**, using both textual content (description, requirements,
company profile) and structured metadata (employment type, presence of a
company logo, etc.).

Built as an internship project. Dataset: **EMSCAD** (Employment Scam Aegean
Dataset), 17,880 real job postings, ~4.8% labeled fraudulent.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9695 | 0.6301 | 0.8960 | 0.7399 | 0.9837 |
| Naive Bayes | 0.9698 | 0.8824 | 0.4335 | 0.5814 | 0.9422 |
| Random Forest | 0.9782 | 1.0000 | 0.5491 | 0.7090 | 0.9874 |
| Linear SVM | 0.9810 | 0.7838 | 0.8382 | 0.8101 | 0.9841 |
| **Logistic Regression (tuned, C=10)** | **0.9799** | **0.7590** | **0.8555** | **0.8043** | **0.9861** |

## Project structure

```
├── notebooks/
│   ├── 01_data_loading_eda.ipynb          # Load data, class balance, missing values, text-length EDA
│   ├── 02_preprocessing.ipynb             # Text cleaning, field combination
│   ├── 03_feature_engineering.ipynb       # TF-IDF / CountVectorizer, train/test split
│   ├── 04_modeling_evaluation.ipynb       # Train & evaluate 4 models, preprocessing comparison
│   ├── 05_tuning_interpretation.ipynb     # GridSearchCV tuning, top-word interpretation
│   └── 06_final_prediction_system.ipynb   # predict_job_posting() function + demo
├── app/
│   ├── app.py                # Streamlit demo app
│   ├── fraud_model.joblib    # Trained model (tuned Logistic Regression)
│   └── tfidf_vectorizer.joblib
├── figures/                  # Generated plots (EDA, confusion matrices, ROC, feature importance)
├── reports/                  # Written report + slide deck
└── data/                     # Dataset goes here (see data/README.md) — not committed
```

Each notebook is self-contained: it loads whatever the previous phase saved
(`data/processed_postings.pkl`, `data/features_bundle.joblib`, or the files in
`app/`), so they're meant to be run in order, 01 → 06. Those intermediate
`.pkl`/`.joblib` cache files aren't committed to the repo (see `.gitignore`) —
only the final `app/fraud_model.joblib` and `app/tfidf_vectorizer.joblib` are,
since the demo app depends on them.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download the dataset per `data/README.md`, then open and run the notebooks in
order (01 through 06) in Jupyter or VS Code.

To try the live demo app once notebook 05 has produced the model files:
```bash
cd app
streamlit run app.py
```

## Approach

1. **EDA** (`01`) — class balance, missing values, text-length patterns by class, structured-field fraud rates.
2. **Preprocessing** (`02`) — text cleaning (HTML/URL/punctuation stripping), missing-value handling.
3. **Feature engineering** (`03`) — TF-IDF and CountVectorizer, compared directly.
4. **Modeling** (`04`) — Logistic Regression, Naive Bayes, Random Forest, Linear SVM; accuracy, precision, recall, F1, ROC-AUC (accuracy alone is misleading on this ~95/5 imbalanced dataset).
5. **Tuning & interpretation** (`05`) — GridSearchCV over Logistic Regression's regularization strength; top TF-IDF coefficients driving fraud vs. legitimate predictions.
6. **Final system** (`06`) — a reusable `predict_job_posting()` function, demoed on real examples, plus a Streamlit app for live use.

## Limitations

- Only ~4.8% of postings are fraudulent — rare fraud patterns can still be missed.
- Trained on 2012–2014 language; scam wording evolves over time.
- Text-only signals — no company/domain verification is performed.
- Best used as one layer of a larger moderation system, alongside human review.

## Dataset credit

Employment Scam Aegean Dataset (EMSCAD), University of the Aegean Laboratory
of Information & Communication Systems Security, distributed via
[Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction).
