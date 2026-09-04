"""
Feature engineering: TF-IDF and CountVectorizer text vectorization.
"""

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

DEFAULT_VECTORIZER_KWARGS = dict(max_features=5000, ngram_range=(1, 2), min_df=3, stop_words="english")


def build_tfidf(text_series, **kwargs):
    """Fit a TF-IDF vectorizer on the given text series."""
    params = {**DEFAULT_VECTORIZER_KWARGS, **kwargs}
    vectorizer = TfidfVectorizer(**params)
    X = vectorizer.fit_transform(text_series)
    return vectorizer, X


def build_count_vectorizer(text_series, **kwargs):
    """Fit a raw-frequency CountVectorizer on the given text series."""
    params = {**DEFAULT_VECTORIZER_KWARGS, **kwargs}
    vectorizer = CountVectorizer(**params)
    X = vectorizer.fit_transform(text_series)
    return vectorizer, X
