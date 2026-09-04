"""
Data loading and text preprocessing utilities for the Fake Job Posting
Detection project.
"""

import re
import pandas as pd

TEXT_FIELDS = ["title", "company_profile", "description", "requirements", "benefits"]


def load_data(path: str) -> pd.DataFrame:
    """Load the raw EMSCAD job postings CSV."""
    df = pd.read_csv(path)
    return df


def clean_text(text: str) -> str:
    """Lowercase, strip HTML tags, URLs, EMSCAD placeholder tokens, and punctuation/digits."""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)                                  # HTML tags
    text = re.sub(r"http\S+|www\.\S+", " ", text)                       # URLs
    text = re.sub(r"#url_\w+#|#email_\w+#|#phone_\w+#", " ", text)      # EMSCAD's anonymized placeholders
    text = re.sub(r"[^a-z\s]", " ", text)                                # punctuation/digits
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_combined_text(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing text fields, concatenate them, and add a cleaned text column."""
    df = df.copy()
    for col in TEXT_FIELDS:
        df[col] = df[col].fillna("")
    df["full_text"] = df[TEXT_FIELDS].agg(" ".join, axis=1)
    df["full_text_clean"] = df["full_text"].apply(clean_text)
    return df


def add_text_length_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `<field>_len` column for each text field (used in EDA)."""
    df = df.copy()
    for col in TEXT_FIELDS:
        if col == "title":
            continue
        df[f"{col}_len"] = df[col].fillna("").apply(len)
    return df
