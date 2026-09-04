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

See `reports/` for the full write-up and slide deck, and `notebooks/` for the
full EDA → preprocessing → modeling → evaluation → interpretation walkthrough.

## Project structure

```
├── src/                    # Reusable pipeline modules
│   ├── preprocessing.py    # Data loading + text cleaning
│   ├── features.py         # TF-IDF / CountVectorizer
│   ├── modeling.py         # Model definitions, training, evaluation
│   ├── predict.py          # Final prediction system
│   └── train_pipeline.py   # End-to-end CLI training script
├── notebooks/
│   └── Fake_Job_Posting_Detection.ipynb   # Full analysis, executed
├── app/
│   ├── app.py               # Streamlit demo app
│   ├── fraud_model.joblib   # Trained model (tuned Logistic Regression)
│   └── tfidf_vectorizer.joblib
├── figures/                 # Generated plots (EDA, confusion matrices, ROC, etc.)
├── reports/                 # Written report + slide deck
└── data/                    # Dataset goes here (see data/README.md)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download the dataset per `data/README.md`, then:

```bash
# Train from scratch
python -m src.train_pipeline --data data/fake_job_postings.csv

# Or open the full notebook
jupyter notebook notebooks/Fake_Job_Posting_Detection.ipynb

# Or run the live demo app
cd app && streamlit run app.py
```

## Approach

1. **EDA** — class balance, missing values, text-length patterns by class, structured-field fraud rates.
2. **Preprocessing** — text cleaning (HTML/URL/punctuation stripping), missing-value handling.
3. **Feature engineering** — TF-IDF and CountVectorizer, compared directly.
4. **Modeling** — Logistic Regression, Naive Bayes, Random Forest, Linear SVM.
5. **Evaluation** — accuracy, precision, recall, F1, ROC-AUC (accuracy alone is misleading on this ~95/5 imbalanced dataset).
6. **Tuning** — GridSearchCV over Logistic Regression's regularization strength.
7. **Interpretation** — top TF-IDF coefficients driving fraud vs. legitimate predictions.
8. **Final system** — a reusable `predict_job_posting()` function + a Streamlit demo app.

## Limitations

- Only ~4.8% of postings are fraudulent — rare fraud patterns can still be missed.
- Trained on 2012–2014 language; scam wording evolves over time.
- Text-only signals — no company/domain verification is performed.
- Best used as one layer of a larger moderation system, alongside human review.

## Dataset credit

Employment Scam Aegean Dataset (EMSCAD), University of the Aegean Laboratory
of Information & Communication Systems Security, distributed via
[Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction).
