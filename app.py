"""Restaurant review sentiment: Streamlit UI around your BernoulliNB model."""
import re
import string
from collections import Counter
from pathlib import Path

import joblib
import nltk
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(page_title="Review Sentiment", page_icon="🍽️", layout="wide")

BASE = Path(__file__).parent
GREEN, RED, SAFFRON = "#2F7D5B", "#C8452F", "#D9A21B"
UNSURE_BELOW = 0.60  # below this confidence, show "Mixed or unsure"

SAMPLES = {
    "Loved it": "The pasta was perfectly cooked and the staff made us feel like regulars. We'll be back every week!",
    "Mixed": "Great ambience and friendly waiter, but the food took 50 minutes and arrived lukewarm.",
    "Disappointed": "Overpriced and bland. The soup was cold, and nobody apologised when we complained.",
}

# -------------------------------------------------------------------- css
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'DM Sans', sans-serif; }
h1, h2, h3, .hero-title, .verdict { font-family: 'DM Serif Display', serif !important; font-weight: 400 !important; }
.block-container { padding-top: 2rem; max-width: 1150px; }
#MainMenu, footer { visibility: hidden; }
.hero-title { font-size: 2.6rem; line-height: 1.1; margin: 0; color: #1B2433; }
.hero-sub { color: #5B6578; font-size: 1.05rem; margin: .4rem 0 1.6rem; max-width: 56ch; }
.verdict-card { background: #fff; border: 1px solid #E3E6EC; border-left: 8px solid var(--c);
  border-radius: 10px; padding: 1.2rem 1.4rem; }
.verdict { font-size: 2.2rem; color: var(--t); margin: 0; }
.verdict-note { color: #5B6578; margin: .2rem 0 0; }
.chip { display: inline-block; padding: 3px 11px; margin: 0 6px 6px 0; border-radius: 999px;
  font-size: .9rem; border: 1px solid var(--c); color: var(--t); background: #fff; }
.stat { background: #fff; border: 1px solid #E3E6EC; border-radius: 10px; padding: .9rem 1.1rem; }
.stat .n { font-family: 'DM Serif Display', serif; font-size: 2rem; line-height: 1.1; }
.stat .l { color: #5B6578; font-size: .9rem; }
.stButton > button { border-radius: 8px; font-weight: 500; }
.stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 500; }
button:focus-visible { outline: 3px solid #1B2433 !important; outline-offset: 2px; }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------- model + nltk data
@st.cache_resource(show_spinner="Loading model…")
def load_assets():
    for pkg in ("stopwords", "wordnet"):
        nltk.download(pkg, quiet=True)
    model = joblib.load(BASE / "sentiment_model.pkl")
    vec = joblib.load(BASE / "binary_vectorizer.pkl")
    return model, vec, set(stopwords.words("english")), WordNetLemmatizer()


model, vectorizer, STOP, LEMMA = load_assets()
NEGATIONS = {"not", "no", "never"}
VOCAB = vectorizer.vocabulary_
# How strongly each word pushes toward Positive (+) or Negative (-) in BernoulliNB
WORD_PUSH = model.feature_log_prob_[1] - model.feature_log_prob_[0]


def preprocess_text(text: str) -> str:
    """Same cleaning as your original app."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if w not in STOP or w in NEGATIONS]
    words = [LEMMA.lemmatize(w) for w in words]
    text = re.sub(r"[^a-zA-Z\s]", "", " ".join(words))
    return re.sub(r"\s+", " ", text).strip()


def predict(review: str):
    """Return (label, confidence, p_positive, cleaned_text)."""
    cleaned = preprocess_text(review)
    p_pos = float(model.predict_proba(vectorizer.transform([cleaned]))[0][list(model.classes_).index(1)])
    label = "Positive" if p_pos >= 0.5 else "Negative"
    return label, max(p_pos, 1 - p_pos), p_pos, cleaned


def driver_words(cleaned: str, n=4):
    """Words in the review the model knows, ranked by pull toward each side."""
    seen = {w: WORD_PUSH[VOCAB[w]] for w in set(cleaned.split()) if w in VOCAB}
    pos = sorted((w for w, s in seen.items() if s > 0), key=lambda w: -seen[w])[:n]
    neg = sorted((w for w, s in seen.items() if s < 0), key=lambda w: seen[w])[:n]
    return pos, neg


def chips(words, color, text_color):
    return "".join(f'<span class="chip" style="--c:{color};--t:{text_color}">{w}</span>' for w in words)


def meter(p_pos: float):
    fig = go.Figure()
    fig.add_bar(x=[p_pos], y=[""], orientation="h", marker_color=GREEN, name="Positive",
                text=f"Positive {p_pos:.0%}", textposition="inside", insidetextanchor="start",
                textfont=dict(color="white", size=14))
    fig.add_bar(x=[1 - p_pos], y=[""], orientation="h", marker_color=RED, name="Negative",
                text=f"Negative {1 - p_pos:.0%}", textposition="inside", insidetextanchor="end",
                textfont=dict(color="white", size=14))
    fig.update_layout(barmode="stack", height=90, showlegend=False, margin=dict(l=0, r=0, t=0, b=0),
                      xaxis=dict(range=[0, 1], visible=False), yaxis=dict(visible=False),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


# ------------------------------------------------------------------ header
st.markdown('<h1 class="hero-title">What are diners really saying?</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Paste a review or upload a file of them. Get the sentiment, how sure the model is, '
    "and the words behind it.</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("About")
    st.caption("Model: **Bernoulli Naive Bayes** on binary word features")
    st.caption("Predicts Positive or Negative. When the model is under 60% sure, the result reads "
               "“Mixed or unsure”.")
    st.divider()
    st.subheader("Try a sample")
    for name, sample in SAMPLES.items():
        if st.button(name, use_container_width=True):
            st.session_state["review_text"] = sample

tab_one, tab_batch = st.tabs(["Single review", "Many reviews (CSV)"])

# --------------------------------------------------------------- single tab
with tab_one:
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        text = st.text_area("Review", key="review_text", height=200, placeholder="Paste a customer review here…")
        run = st.button("Analyze review", type="primary")

    with right:
        if run and text.strip():
            label, conf, p_pos, cleaned = predict(text)
            if conf < UNSURE_BELOW:
                shown, bar, ink = "Mixed or unsure", SAFFRON, "#8A6200"
            else:
                shown, bar, ink = label, (GREEN if label == "Positive" else RED), (GREEN if label == "Positive" else RED)
            st.markdown(
                f'<div class="verdict-card" style="--c:{bar};--t:{ink}"><p class="verdict">{shown}</p>'
                f'<p class="verdict-note">Leans {label.lower()}, {conf:.0%} confidence</p></div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(meter(p_pos), use_container_width=True, config={"displayModeBar": False})
            pos, neg = driver_words(cleaned)
            if pos or neg:
                st.markdown("**Words that moved the result**")
                if pos:
                    st.markdown(chips(pos, GREEN, GREEN), unsafe_allow_html=True)
                if neg:
                    st.markdown(chips(neg, RED, RED), unsafe_allow_html=True)
            else:
                st.caption("None of these words were in the model's vocabulary, so treat this result with care.")
        elif run:
            st.warning("Paste a review first, then select Analyze review.")
        else:
            st.info("Your result will appear here. Pick a sample in the sidebar to try it out.")

# ---------------------------------------------------------------- batch tab
with tab_batch:
    file = st.file_uploader("Upload a CSV with a column of reviews", type="csv")
    if file is None:
        st.info("No file yet. Upload a CSV to see sentiment across all your reviews.")
    else:
        df = pd.read_csv(file)
        guess = next((i for i, c in enumerate(df.columns) if "review" in c.lower() or "text" in c.lower()), 0)
        col = st.selectbox("Which column holds the review text?", df.columns, index=guess)
        if st.button("Analyze all reviews", type="primary"):
            with st.spinner(f"Analyzing {len(df):,} reviews…"):
                texts = df[col].fillna("").astype(str)
                cleaned = texts.apply(preprocess_text)
                p_pos = model.predict_proba(vectorizer.transform(cleaned))[:, list(model.classes_).index(1)]
                res = df.copy()
                res["sentiment"] = ["Positive" if p >= 0.5 else "Negative" for p in p_pos]
                res["confidence"] = [round(max(p, 1 - p), 3) for p in p_pos]
                res["_clean"] = cleaned
                st.session_state["batch"] = res

        res = st.session_state.get("batch")
        if res is not None:
            total = len(res)
            counts = res["sentiment"].value_counts()
            c = st.columns(3)
            c[0].markdown(f'<div class="stat"><div class="n">{total:,}</div><div class="l">Reviews</div></div>', unsafe_allow_html=True)
            for box, k, colr in zip(c[1:], ["Positive", "Negative"], [GREEN, RED]):
                n = int(counts.get(k, 0))
                box.markdown(f'<div class="stat"><div class="n" style="color:{colr}">{n / total:.0%}</div>'
                             f'<div class="l">{k} ({n:,})</div></div>', unsafe_allow_html=True)

            a, b = st.columns(2, gap="large")
            with a:
                st.subheader("Sentiment mix")
                fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=.55, sort=False,
                                       marker_colors=[GREEN if k == "Positive" else RED for k in counts.index]))
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            with b:
                st.subheader("Words that come up most")
                pick = st.radio("In", ["Negative", "Positive"], horizontal=True, label_visibility="collapsed")
                words = Counter(w for t in res.loc[res["sentiment"] == pick, "_clean"] for w in t.split()
                                if w not in NEGATIONS).most_common(8)
                if words:
                    wf = go.Figure(go.Bar(x=[n for _, n in words], y=[w for w, _ in words], orientation="h",
                                          marker_color=GREEN if pick == "Positive" else RED))
                    wf.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(autorange="reversed"))
                    st.plotly_chart(wf, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.caption(f"No {pick.lower()} reviews in this file.")

            st.subheader("Results")
            show = st.multiselect("Show", ["Positive", "Negative"], default=["Positive", "Negative"])
            out = res.drop(columns="_clean")
            st.dataframe(out[out["sentiment"].isin(show)], use_container_width=True, height=320)
            st.download_button("Download results (CSV)", out.to_csv(index=False).encode(),
                               "sentiment_results.csv", "text/csv")
