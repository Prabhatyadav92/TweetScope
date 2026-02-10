import streamlit as st
import joblib
import numpy as np

# -----------------------------
# Session State
# -----------------------------
if "example" not in st.session_state:
    st.session_state.example = ""

# -----------------------------
# Load Model & Vectorizer (cached)
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("trained_model.sav")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="TweetScope | Sentiment Analysis",
    page_icon="🐦",
    layout="centered"
)

# -----------------------------
# Custom CSS (Pure Streamlit)
# -----------------------------
st.markdown("""
<style>
:root {
    --primary: #1DA1F2;
    --positive: #00C853;
    --negative: #FF5252;
    --neutral: #FFC107;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f5f7fa, #e4e7eb);
}

.card {
    background: white;
    padding: 2rem;
    border-radius: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    margin-top: 1rem;
}

.stButton>button {
    background: var(--primary);
    color: white;
    border-radius: 50px;
    padding: 12px 24px;
    font-weight: 600;
    border: none;
    width: 100%;
}

.stButton>button:hover {
    background: #0d8bd9;
    transform: translateY(-2px);
}

.confidence-bar {
    height: 12px;
    border-radius: 10px;
    background: #e0e0e0;
    overflow: hidden;
    margin-top: 10px;
}

.confidence-fill {
    height: 100%;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <h1 style="text-align:center; color:#1DA1F2;">🐦 TweetScope</h1>
    <p style="text-align:center; color:#657786;">
    AI-powered Twitter Sentiment Analysis
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("ℹ️ About")
    st.write(
        "TweetScope analyzes the sentiment of tweets using "
        "Machine Learning and Natural Language Processing."
    )
    st.markdown("**Tech Stack**")
    st.markdown("- Python\n- TF-IDF\n- ML Classifier\n- Streamlit")

# -----------------------------
# Main Card
# -----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("📝 Analyze Tweet Sentiment")

tweet = st.text_area(
    "Tweet Input",
    value=st.session_state.example,
    placeholder="What's happening?",
    height=150,
    label_visibility="collapsed"
)

# Example tweets
st.caption("Try examples:")
c1, c2, c3 = st.columns(3)

with c1:
    if st.button("😊 Positive"):
        st.session_state.example = (
            "Just had the best coffee ever! This new cafe is amazing!"
        )
        st.experimental_rerun()

with c2:
    if st.button("😠 Negative"):
        st.session_state.example = (
            "Worst customer service ever. Completely disappointed."
        )
        st.experimental_rerun()

with c3:
    if st.button("😐 Neutral"):
        st.session_state.example = (
            "The weather forecast predicts rain tomorrow."
        )
        st.experimental_rerun()

analyze = st.button("🔍 Analyze Sentiment", type="primary")

# -----------------------------
# Prediction
# -----------------------------
if analyze:
    if not tweet.strip():
        st.warning("⚠️ Please enter a tweet.")
    else:
        with st.spinner("Analyzing sentiment..."):
            tweet_vec = vectorizer.transform([tweet])
            prediction = model.predict(tweet_vec)[0]

            if hasattr(model, "predict_proba"):
                confidence = np.max(model.predict_proba(tweet_vec)) * 100
            else:
                confidence = 90.0

        st.markdown("---")

        if prediction == 1:
            sentiment, emoji, color = "Positive", "😊", "var(--positive)"
        elif prediction == 0:
            sentiment, emoji, color = "Negative", "😠", "var(--negative)"
        else:
            sentiment, emoji, color = "Neutral", "😐", "var(--neutral)"

        st.markdown(
            f"<h2 style='text-align:center;'>{emoji} {sentiment} Sentiment</h2>",
            unsafe_allow_html=True
        )

        st.write(f"**Confidence:** {confidence:.1f}%")

        st.markdown(
            f"""
            <div class="confidence-bar">
                <div class="confidence-fill"
                     style="width:{confidence}%; background:{color};"></div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption("Your Tweet:")
        st.info(tweet)

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#657786;'>"
    "Built with ❤️ by <b>Prabhat Yadav</b> | Streamlit"
    "</div>",
    unsafe_allow_html=True
)
