"""
personality_model.py
---------------------
Trains one binary classifier per Big Five trait (multi-output via five
separate models, since each trait is a distinct y/n prediction, not a
single multi-class problem).

Uses Logistic Regression - a strong, standard baseline for TF-IDF text
classification, and importantly interpretable: coefficients directly show
which words push a prediction toward "high" or "low" on a trait, which
matters for explaining predictions rather than just producing them.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from text_processor import clean_text, build_vectorizer

TRAITS = ["EXT", "NEU", "AGR", "CON", "OPN"]
TRAIT_NAMES = {
    "EXT": "Extraversion",
    "NEU": "Neuroticism",
    "AGR": "Agreeableness",
    "CON": "Conscientiousness",
    "OPN": "Openness",
}

MODEL_DIR = "models"
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")
MODEL_PATH_TEMPLATE = os.path.join(MODEL_DIR, "model_{trait}.joblib")

RANDOM_SEED = 42
TEST_SIZE = 0.2


def load_essays_data(path="data/essays.csv"):
    """Loads the essays dataset. Works with both the real Pennebaker & King
    dataset and our synthetic generator's output - same schema."""
    df = pd.read_csv(path, encoding="latin-1")  # real dataset is often latin-1 encoded

    required = {"TEXT"} | {f"c{t}" for t in TRAITS}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset missing expected columns {required}. Got: {list(df.columns)}")

    return df


def train_all_models(df):
    """
    Trains one Logistic Regression classifier per trait.

    Returns:
        vectorizer: fitted TF-IDF vectorizer (shared across all traits)
        models: dict of {trait: fitted LogisticRegression}
        eval_results: dict of {trait: classification report string}
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    cleaned_text = df["TEXT"].apply(clean_text)

    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(cleaned_text)

    models = {}
    eval_results = {}

    for trait in TRAITS:
        y = (df[f"c{trait}"].str.lower() == "y").astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
        )

        model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=["Low", "High"], zero_division=0)

        print(f"\n=== {TRAIT_NAMES[trait]} ({trait}) — accuracy: {acc:.3f} ===")
        print(report)

        models[trait] = model
        eval_results[trait] = {"accuracy": acc, "report": report}

    return vectorizer, models, eval_results


def save_models(vectorizer, models):
    joblib.dump(vectorizer, VECTORIZER_PATH)
    for trait, model in models.items():
        joblib.dump(model, MODEL_PATH_TEMPLATE.format(trait=trait))
    print(f"\nAll models saved to {MODEL_DIR}/")


def load_models():
    """Loads the vectorizer and all five trait models. Raises FileNotFoundError
    with a clear message if training hasn't been run yet."""
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            "No trained models found. Run `python main.py` first to train them."
        )

    vectorizer = joblib.load(VECTORIZER_PATH)
    models = {trait: joblib.load(MODEL_PATH_TEMPLATE.format(trait=trait)) for trait in TRAITS}
    return vectorizer, models


def predict_personality(vectorizer, models, text):
    """
    Predicts Big Five trait probabilities for a piece of text (e.g. resume content).

    Returns:
        dict of {trait: {"prediction": "High"/"Low", "confidence": float}}
    """
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])

    results = {}
    for trait, model in models.items():
        proba = model.predict_proba(X)[0]
        pred_class = model.predict(X)[0]
        confidence = proba[pred_class]

        results[trait] = {
            "trait_name": TRAIT_NAMES[trait],
            "prediction": "High" if pred_class == 1 else "Low",
            "confidence": float(confidence),
            "high_probability": float(proba[1]),
        }

    return results


def get_top_words_for_trait(vectorizer, model, trait, top_n=10):
    """Returns the words most strongly associated with a HIGH prediction for
    a given trait, based on logistic regression coefficients. Useful for
    explaining predictions, not just stating them."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefficients = model.coef_[0]

    top_indices = np.argsort(coefficients)[-top_n:][::-1]
    return [(feature_names[i], float(coefficients[i])) for i in top_indices]
