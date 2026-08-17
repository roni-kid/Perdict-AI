"""
text_processor.py
------------------
Text cleaning and TF-IDF feature extraction shared by both training
(on essays) and inference (on resumes/CVs).

Using the SAME vectorizer object for both training and inference is critical -
it's fit once on training data and reused (not refit) on new text, otherwise
the feature space won't match and predictions will be garbage.
"""

import re

from sklearn.feature_extraction.text import TfidfVectorizer

MAX_FEATURES = 3000


def clean_text(text):
    """Basic cleaning: lowercase, strip non-letter characters, collapse whitespace.
    Deliberately light-touch - TF-IDF with stopword removal handles a lot of
    the heavy lifting, and personality-relevant word choice can be subtle
    enough that aggressive cleaning (e.g. removing all punctuation/case) risks
    losing signal.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_vectorizer():
    """Creates a fresh TF-IDF vectorizer. Call fit_transform on TRAINING data
    only, then use .transform() (not fit_transform) on any new text."""
    return TfidfVectorizer(
        max_features=MAX_FEATURES,
        stop_words="english",
        ngram_range=(1, 2),  # unigrams + bigrams - captures short phrases, not just single words
        min_df=2,  # ignore words that appear in fewer than 2 documents (likely noise)
    )
