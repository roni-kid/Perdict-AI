"""
app.py
------
Streamlit UI for Perdict AI. Upload a resume/CV
(PDF, DOCX, or TXT) and get Big Five personality trait predictions, or
paste text directly.

Run:
    streamlit run app.py

Requires models/ to already contain trained models - run `python main.py`
first if they don't exist yet.
"""

import os

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from personality_model import load_models, predict_personality, get_top_words_for_trait, TRAITS, TRAIT_NAMES
from resume_parser import extract_resume_text

st.set_page_config(page_title="Personality Prediction from CV", page_icon="🧬", layout="wide")


@st.cache_resource
def get_models():
    try:
        return load_models()
    except FileNotFoundError:
        return None, None


def render_radar_chart(results):
    """Renders a radar/spider chart of the five trait scores."""
    traits = list(results.keys())
    labels = [results[t]["trait_name"] for t in traits]
    scores = [results[t]["high_probability"] * 100 for t in traits]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores + [scores[0]],  # close the loop
        theta=labels + [labels[0]],
        fill="toself",
        name="Personality profile",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        title="Big Five trait profile (% likelihood of 'High')",
    )
    return fig


def render_results(results, vectorizer, models):
    st.subheader("Predicted personality profile")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.plotly_chart(render_radar_chart(results), use_container_width=True)

    with col2:
        for trait, r in results.items():
            st.markdown(f"**{r['trait_name']}**")
            st.progress(r["high_probability"])
            st.caption(f"{r['prediction']} ({r['confidence']:.0%} confidence)")

    with st.expander("Why these predictions? (top influencing words per trait)"):
        for trait in TRAITS:
            top_words = get_top_words_for_trait(vectorizer, models[trait], trait, top_n=8)
            words_str = ", ".join(w for w, _ in top_words)
            st.markdown(f"**{TRAIT_NAMES[trait]}** — words pushing toward *High*: {words_str}")


def main():
    st.title("🧬 Perdict AI")
    st.caption("Predicts Big Five personality traits from resume/CV text")

    vectorizer, models = get_models()

    if vectorizer is None:
        st.error(
            "No trained models found. Run `python main.py` first to train "
            "and save the models, then reload this page."
        )
        st.stop()

    st.warning(
        "⚠️ **Important limitation:** this model is trained on personal essays "
        "(free-flowing prose about someone's thoughts/experiences), not resumes. "
        "Resumes are short, structured, and impersonal by design — very "
        "different from the training data. Treat predictions here as a rough, "
        "experimental signal, not a validated personality assessment. This "
        "should never be used as the sole basis for a hiring decision.",
        icon="⚠️",
    )

    tab_upload, tab_paste = st.tabs(["📄 Upload a resume/CV", "✍️ Paste text directly"])

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload a resume or CV", type=["pdf", "docx", "txt"]
        )

        if uploaded_file is not None:
            try:
                text = extract_resume_text(uploaded_file, filename=uploaded_file.name)
            except Exception as e:
                st.error(f"Couldn't extract text from that file: {e}")
                st.stop()

            if not text.strip():
                st.warning("No extractable text found in this file (it may be a scanned image).")
                st.stop()

            with st.expander("Extracted text (preview)"):
                st.text(text[:2000] + ("..." if len(text) > 2000 else ""))

            if st.button("Analyze this resume", key="analyze_upload"):
                with st.spinner("Analyzing..."):
                    results = predict_personality(vectorizer, models, text)
                render_results(results, vectorizer, models)

    with tab_paste:
        pasted_text = st.text_area(
            "Paste resume/CV text here", height=250,
            placeholder="Paste the text content of a resume or CV..."
        )

        if st.button("Analyze this text", key="analyze_paste"):
            if not pasted_text.strip():
                st.warning("Please paste some text first.")
            else:
                with st.spinner("Analyzing..."):
                    results = predict_personality(vectorizer, models, pasted_text)
                render_results(results, vectorizer, models)


if __name__ == "__main__":
    main()
