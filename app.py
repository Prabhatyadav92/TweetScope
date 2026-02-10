import streamlit as st
import joblib
import numpy as np
import time

# -----------------------------
# Session State
# -----------------------------
if "example" not in st.session_state:
    st.session_state.example = ""
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None

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
    layout="centered",
    initial_sidebar_state="collapsed"  # Sidebar collapsed by default
)

# -----------------------------
# Custom CSS (Enhanced)
# -----------------------------
st.markdown("""
<style>
:root {
    --primary: #1DA1F2;
    --positive: #00C853;
    --negative: #FF5252;
    --neutral: #FFC107;
    --refresh: #66757F;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f5f7fa, #e4e7eb);
}

.main-card {
    background: white;
    padding: 2rem;
    border-radius: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    margin-top: 1rem;
    margin-bottom: 2rem;
}

.result-card {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 5px solid var(--primary);
    margin: 1.5rem 0;
}

.stButton>button[data-testid="baseButton-primary"] {
    background: var(--primary);
    color: white;
    border-radius: 50px;
    padding: 12px 24px;
    font-weight: 600;
    border: none;
    width: 100%;
    transition: all 0.3s ease;
}

.stButton>button[data-testid="baseButton-primary"]:hover {
    background: #0d8bd9;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(29, 161, 242, 0.3);
}

.stButton>button[data-testid="baseButton-secondary"] {
    background: var(--refresh);
    color: white;
    border-radius: 50px;
    padding: 8px 20px;
    font-weight: 600;
    border: none;
    width: auto;
    transition: all 0.3s ease;
}

.stButton>button[data-testid="baseButton-secondary"]:hover {
    background: #546371;
    transform: translateY(-2px);
}

.example-buttons .stButton>button {
    background: #E8F5FE;
    color: var(--primary);
    border: 1px solid #AAB8C2;
    border-radius: 25px;
    padding: 8px 16px;
    font-weight: 500;
    width: 100%;
    transition: all 0.3s ease;
}

.example-buttons .stButton>button:hover {
    background: #D4E8FF;
    transform: translateY(-1px);
}

.confidence-bar {
    height: 16px;
    border-radius: 10px;
    background: #e0e0e0;
    overflow: hidden;
    margin: 15px 0;
    position: relative;
}

.confidence-fill {
    height: 100%;
    border-radius: 10px;
    transition: width 1s ease-in-out;
}

.confidence-text {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: white;
    font-weight: bold;
    font-size: 12px;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
}

.sentiment-emoji {
    font-size: 3rem;
    text-align: center;
    margin: 10px 0;
}

.sentiment-text {
    font-size: 1.8rem;
    font-weight: bold;
    text-align: center;
    margin: 10px 0;
}

.tweet-display {
    background: #F7F9FA;
    border-radius: 12px;
    padding: 15px;
    border-left: 4px solid #AAB8C2;
    font-style: italic;
    margin: 15px 0;
}

.footer {
    text-align: center;
    padding: 20px;
    color: #657786;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div style="text-align:center; margin-bottom: 30px;">
        <h1 style="color:#1DA1F2; margin-bottom: 10px;">
            🐦 TweetScope
            <span style="font-size: 1rem; color:#657786; font-weight: normal;">
                | Real-time Sentiment Analysis
            </span>
        </h1>
        <p style="color:#657786; font-size: 1.1rem;">
        Understand the sentiment behind tweets with AI-powered analysis
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Main Card
# -----------------------------
st.markdown("<div class='main-card'>", unsafe_allow_html=True)

# Refresh button at top right
col1, col2 = st.columns([5, 1])
with col1:
    st.subheader("📝 Analyze Tweet Sentiment")
with col2:
    if st.button("🔄 Refresh", key="refresh_top", type="secondary"):
        st.session_state.example = ""
        st.session_state.last_analysis = None
        st.experimental_rerun()

# Tweet input
tweet = st.text_area(
    "Tweet Input",
    value=st.session_state.example,
    placeholder="What's happening? 🤔\n\nPaste a tweet or type your message here...",
    height=120,
    label_visibility="collapsed",
    key="tweet_input"
)

# Example tweets
st.markdown("<div class='example-buttons'>", unsafe_allow_html=True)
st.caption("Try example tweets:")

c1, c2, c3, c4 = st.columns(4)

examples = {
    "😊 Positive": "Just had the best coffee ever! This new cafe is amazing! The barista was so friendly and the atmosphere is perfect for working. Highly recommend! #coffeelovers",
    "😠 Negative": "Worst customer service ever. Waited 2 hours on hold just to be disconnected. Completely disappointed with this company. Never buying again! #customerservicefail",
    "😐 Neutral": "The weather forecast predicts rain tomorrow afternoon with a high of 68°F. Make sure to carry an umbrella if you're heading out.",
    "🤔 Mixed": "The movie had amazing visual effects but the storyline was predictable. Great for action fans but don't expect Oscar-winning writing."
}

with c1:
    if st.button("😊 Positive"):
        st.session_state.example = examples["😊 Positive"]
        st.experimental_rerun()

with c2:
    if st.button("😠 Negative"):
        st.session_state.example = examples["😠 Negative"]
        st.experimental_rerun()

with c3:
    if st.button("😐 Neutral"):
        st.session_state.example = examples["😐 Neutral"]
        st.experimental_rerun()

with c4:
    if st.button("🤔 Mixed"):
        st.session_state.example = examples["🤔 Mixed"]
        st.experimental_rerun()

st.markdown("</div>", unsafe_allow_html=True)

# Analyze button
col1, col2 = st.columns([3, 1])
with col1:
    analyze = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
with col2:
    if st.button("🗑️ Clear", type="secondary", use_container_width=True):
        st.session_state.example = ""
        st.session_state.last_analysis = None
        st.experimental_rerun()

# -----------------------------
# Prediction Logic
# -----------------------------
def analyze_sentiment(text):
    """Analyze sentiment and return results"""
    tweet_vec = vectorizer.transform([text])
    prediction = model.predict(tweet_vec)[0]
    
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(tweet_vec)[0]
        confidence = np.max(proba) * 100
        confidence_scores = {
            0: proba[0] * 100 if len(proba) > 0 else 0,
            1: proba[1] * 100 if len(proba) > 1 else 0,
            2: proba[2] * 100 if len(proba) > 2 else 0
        }
    else:
        confidence = 90.0
        confidence_scores = {0: 0, 1: 0, 2: 0}
    
    # Map prediction to sentiment
    sentiment_map = {
        1: {"label": "Positive", "emoji": "😊", "color": "var(--positive)"},
        0: {"label": "Negative", "emoji": "😠", "color": "var(--negative)"},
        2: {"label": "Neutral", "emoji": "😐", "color": "var(--neutral)"}
    }
    
    result = sentiment_map.get(prediction, sentiment_map[2])
    result["confidence"] = confidence
    result["confidence_scores"] = confidence_scores
    result["prediction"] = prediction
    
    return result

# -----------------------------
# Display Results
# -----------------------------
if analyze or st.session_state.last_analysis:
    if not tweet.strip():
        st.warning("⚠️ Please enter a tweet to analyze.")
    else:
        if analyze:
            with st.spinner("🔮 Analyzing sentiment..."):
                time.sleep(0.5)  # Small delay for better UX
                result = analyze_sentiment(tweet)
                st.session_state.last_analysis = result
        else:
            result = st.session_state.last_analysis
        
        # Display results
        st.markdown("---")
        
        # Result card
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        
        # Sentiment display
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                f"<div class='sentiment-emoji'>{result['emoji']}</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div class='sentiment-text' style='color:{result['color']};'>"
                f"{result['label']} Sentiment</div>",
                unsafe_allow_html=True
            )
        
        # Confidence bar with animation
        st.markdown(
            f"""
            <div style="margin: 20px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span>Confidence Score</span>
                    <span style="font-weight: bold; color:{result['color']};">{result['confidence']:.1f}%</span>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width:{result['confidence']}%; background:{result['color']};">
                        <div class="confidence-text">{result['confidence']:.1f}%</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Confidence breakdown (if available)
        if result.get('confidence_scores'):
            st.caption("Confidence Breakdown:")
            cols = st.columns(3)
            sentiments = [
                ("Negative", "var(--negative)", 0),
                ("Positive", "var(--positive)", 1),
                ("Neutral", "var(--neutral)", 2)
            ]
            
            for idx, (label, color, pred_key) in enumerate(sentiments):
                score = result['confidence_scores'].get(pred_key, 0)
                with cols[idx]:
                    st.metric(
                        label=label,
                        value=f"{score:.1f}%",
                        delta=None
                    )
        
        # Original tweet display
        st.markdown("<div class='tweet-display'>", unsafe_allow_html=True)
        st.write("**Your Tweet:**")
        st.write(tweet)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Refresh button in results
        if st.button("🔄 Analyze Another Tweet", type="secondary", use_container_width=True):
            st.session_state.example = ""
            st.session_state.last_analysis = None
            st.experimental_rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    """
    <div class="footer">
        <div style="margin-bottom: 10px;">
            Built with ❤️ by <b>Prabhat Yadav</b> | Powered by Streamlit
        </div>
        <div style="font-size: 0.8rem; color: #AAB8C2;">
            TweetScope v1.0 | Real-time Sentiment Analysis Tool
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
