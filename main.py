"""
main.py
-------
Entry point for Perdict AI: trains all five Big Five trait models on the
essays dataset, evaluates them, and saves them. Run this before using
app.py or predicting on real resumes.

Usage:
    python generate_sample_data.py   # one-time, if not using the real Essays dataset
    python main.py
"""

from personality_model import load_essays_data, train_all_models, save_models, TRAIT_NAMES


def main():
    print("Loading essays dataset...")
    df = load_essays_data()
    print(f"Loaded {len(df)} labeled text samples\n")

    print("Training models for all five traits...")
    vectorizer, models, eval_results = train_all_models(df)

    save_models(vectorizer, models)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for trait, name in TRAIT_NAMES.items():
        acc = eval_results[trait]["accuracy"]
        print(f"  {name:20s} accuracy: {acc:.3f}")

    print("\nDone. Run `streamlit run app.py` to test on real resumes/CVs.")


if __name__ == "__main__":
    main()
