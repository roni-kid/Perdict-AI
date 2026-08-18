# Perdict AI

Crixsoft Solution AI Internship — Project 3

Predicts Big Five (OCEAN) personality traits — Extraversion, Neuroticism,
Agreeableness, Conscientiousness, Openness — from resume/CV text, using
models trained on a labeled personal-essays dataset.

## Important limitation (read this first)

This model is trained on **personal essays** (free-flowing personal writing),
not resumes. Resumes are short, structured, and impersonal by design — very
different from the training data's style. Predictions on resume text should
be treated as a **rough, experimental signal**, not a validated personality
assessment. This should never be the sole basis for a hiring decision — the
app itself displays this warning prominently.

## Setup

```bash
pip install pandas numpy scikit-learn nltk PyPDF2 python-docx joblib streamlit plotly
```

### Option A — Real dataset (recommended)
1. Download the Essays (Pennebaker & King) Big Five dataset — search
   "essays.csv big five personality" or find it via the
   [jkwieser/personality-prediction-from-text](https://github.com/jkwieser/personality-prediction-from-text)
   repo, which documents and links the source.
2. Save it as `data/essays.csv` with columns: `TEXT, cEXT, cNEU, cAGR, cCON, cOPN`
   (labels are `y`/`n` per trait).
3. Run `python main.py` — no code changes needed, same schema.

### Option B — Synthetic data (for a quick test run)
```bash
python generate_sample_data.py   # generates data/essays.csv
python main.py
```

> The synthetic generator produces artificially clean word-trait
> associations, so accuracy will look better (often ~100%) than it would on
> real data. Use the real Essays dataset before drawing conclusions or
> sharing results.

## Running the app

```bash
streamlit run app.py
```

Upload a PDF, DOCX, or TXT resume — or paste text directly — and get:
- A radar chart of all five trait scores
- Per-trait High/Low prediction with confidence
- An explanation view showing which words most influenced each prediction

## Project structure

```
personality-prediction-ai/
├── data/
│   └── essays.csv                # TEXT, cEXT, cNEU, cAGR, cCON, cOPN
├── models/                       # saved vectorizer + 5 trait models (.joblib)
├── generate_sample_data.py       # synthetic data generator (for testing)
├── text_processor.py             # text cleaning + TF-IDF vectorization
├── personality_model.py          # trains/evaluates/predicts for all 5 traits
├── resume_parser.py              # extracts text from PDF/DOCX/TXT resumes
├── main.py                       # trains all 5 models, entry point
├── app.py                        # Streamlit UI
└── README.md
```

## How it works

1. **Training** — each of the five Big Five traits gets its own binary
   Logistic Regression classifier (High/Low), trained on TF-IDF features
   from the essays dataset. Five separate models rather than one multi-output
   model, since each trait is an independent y/n prediction.
2. **Inference on resumes** — `resume_parser.py` extracts plain text from an
   uploaded PDF/DOCX/TXT, which is then cleaned and vectorized using the
   *same* TF-IDF vectorizer fit during training (critical — refitting on new
   text would produce a mismatched feature space).
3. **Explainability** — Logistic Regression coefficients directly show which
   words push a prediction toward "High" for each trait, so predictions come
   with a "why," not just a score.

## Key design decisions

- **Five separate binary classifiers, not one multi-label model** — keeps
  each trait's decision boundary independent and interpretable, and makes
  per-trait evaluation straightforward.
- **Logistic Regression over more complex models** — for TF-IDF text
  features on a modestly-sized dataset, it's a strong baseline and, more
  importantly, its coefficients are directly interpretable — you can show
  *which words* drove a prediction, not just the prediction itself.
- **Same vectorizer object for train and inference** — fit once on training
  data, reused (never refit) at inference time, so the feature space stays
  consistent between essays and resumes.
- **Explicit limitation warning in the UI** — the domain mismatch between
  essays (training data) and resumes (real-world input) is a genuine,
  known limitation, not a hidden caveat.

## Dataset credit

Essays dataset originally collected by James Pennebaker and Laura King,
widely used as a benchmark for Big Five personality prediction from text.

