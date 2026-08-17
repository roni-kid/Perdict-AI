"""
generate_sample_data.py
------------------------
Generates a synthetic text + Big Five labels dataset that mirrors the schema
of the real "Essays" (Pennebaker & King) Big Five dataset - free text plus
binary y/n labels for each of the five traits: EXT, NEU, AGR, CON, OPN.

To use the REAL dataset instead:
1. Download the Essays-Big5 dataset (search "essays.csv big five Pennebaker")
   - commonly distributed as essays.csv with columns:
     TEXT, cEXT, cNEU, cAGR, cCON, cOPN (y/n labels)
2. Save as data/essays.csv
3. Skip this script entirely - everything downstream reads the same columns.

Run:
    python generate_sample_data.py
"""

import os

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_SAMPLES = 600

TRAITS = ["EXT", "NEU", "AGR", "CON", "OPN"]

# Word banks loosely associated with each trait (high-trait language),
# used only to generate plausible-looking synthetic text - not a real
# lexicon-based model, just enough to test the pipeline end-to-end today.
TRAIT_WORDS = {
    "EXT": ["party", "friends", "outgoing", "energetic", "social", "talkative",
            "excited", "fun", "people", "crowd"],
    "NEU": ["anxious", "worried", "stressed", "nervous", "overwhelmed", "upset",
            "moody", "insecure", "tense", "sad"],
    "AGR": ["kind", "helpful", "caring", "trust", "cooperative", "warm",
            "generous", "considerate", "gentle", "supportive"],
    "CON": ["organized", "plan", "responsible", "deadline", "disciplined",
            "careful", "thorough", "reliable", "goals", "efficient"],
    "OPN": ["curious", "creative", "imagine", "explore", "ideas", "art",
            "philosophy", "novel", "abstract", "wonder"],
}

FILLER_WORDS = ["the", "and", "was", "today", "went", "then", "really", "just",
                "think", "feel", "about", "some", "time", "day", "when", "that"]


def generate_essay(rng, trait_flags):
    """Builds a synthetic essay biased toward words for traits marked True."""
    words = []
    for trait, is_high in trait_flags.items():
        if is_high:
            words.extend(rng.choice(TRAIT_WORDS[trait], size=8, replace=True))
    words.extend(rng.choice(FILLER_WORDS, size=25, replace=True))
    rng.shuffle(words)
    return " ".join(words)


def generate_sample_data():
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    for _ in range(N_SAMPLES):
        trait_flags = {trait: rng.random() > 0.5 for trait in TRAITS}
        text = generate_essay(rng, trait_flags)
        row = {"TEXT": text}
        for trait in TRAITS:
            row[f"c{trait}"] = "y" if trait_flags[trait] else "n"
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_sample_data()
    df.to_csv("data/essays.csv", index=False)
    print(f"Generated {len(df)} synthetic essays with Big Five labels")
    for trait in TRAITS:
        pct = (df[f"c{trait}"] == "y").mean()
        print(f"  {trait}: {pct:.1%} labeled 'y'")
    print("Saved to data/essays.csv")
