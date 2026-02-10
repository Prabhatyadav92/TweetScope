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
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Custom CSS (Enhanced for Binary Classification)
# -----------------------------
st.markdown("""
<style>
:root {
    --primary: #1DA1F2;
    --positive: #00C853;
    --negative: #FF5252;
    --refresh: #66757F;
    --bg-gradient-start: #f5f7fa;
    --bg-gradient-end: #e4e7eb;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, var(--bg-gradient-start), var(--bg-gradient-end));
    min-height: 100vh;
}

.main-card {
    background: white;
    padding: 2.5rem;
    border-radius: 20px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.1);
    margin-top: 1.5rem;
    margin-bottom: 2rem;
    border: 1px solid rgba(0,0,0,0.05);
}

.result-card {
    background: linear-gradient(135deg, #f8f9fa, #ffffff);
    padding: 2rem;
    border-radius: 16px;
    margin: 2rem 0;
    border: 1px solid #e1e8ed;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
}

.positive-card {
    border-left: 6px solid var(--positive);
    background: linear-gradient(135deg, #f0fff4, #ffffff);
}

.negative-card {
    border-left: 6px solid var(--negative);
    background: linear-gradient(135deg, #fff5f5, #ffffff);
}

.stButton>button[data-testid="baseButton-primary"] {
    background: var(--primary);
    color: white;
    border-radius: 50px;
    padding: 14px 28px;
    font-weight: 600;
    border: none;
    width: 100%;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    font-size: 1.1rem;
    letter-spacing: 0.3px;
}

.stButton>button[data-testid="baseButton-primary"]:hover {
    background: #0d8bd9;
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(29, 161, 242, 0.4);
}

.stButton>button[data-testid="baseButton-secondary"] {
    background: var(--refresh);
    color: white;
    border-radius: 50px;
    padding: 10px 24px;
    font-weight: 600;
    border: none;
    transition: all 0.3s ease;
}

.stButton>button[data-testid="baseButton-secondary"]:hover {
    background: #546371;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102, 117, 127, 0.3);
}

.example-buttons .stButton>button {
    background: #E8F5FE;
    color: var(--primary);
    border: 2px solid #AAB8C2;
    border-radius: 30px;
    padding: 10px 20px;
    font-weight: 600;
    width: 100%;
    transition: all 0.3s ease;
    font-size: 0.95rem;
}

.example-buttons .stButton>button:hover {
    background: #D4E8FF;
    transform: translateY(-2px);
    border-color: var(--primary);
}

.confidence-container {
    margin: 25px 0;
    padding: 20px;
    background: #f8fafc;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}

.confidence-bar {
    height: 20px;
    border-radius: 10px;
    background: #e2e8f0;
    overflow: hidden;
    margin: 15px 0;
    position: relative;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
}

.confidence-fill {
    height: 100%;
    border-radius: 10px;
    transition: width 1.2s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
}

.confidence-text {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: white;
    font-weight: bold;
    font-size: 13px;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
    letter-spacing: 0.5px;
}

.sentiment-display {
    text-align: center;
    padding: 25px;
    border-radius: 16px;
    margin: 25px 0;
    animation: fadeIn 0.8s ease-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.sentiment-emoji {
    font-size: 4rem;
    margin-bottom: 15px;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.1); }
    100% { transform: scale(1); }
}

.sentiment-text {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 10px 0;
    letter-spacing: 1px;
}

.tweet-display {
    background: #F7F9FA;
    border-radius: 12px;
    padding: 20px;
    margin: 25px 0;
    border: 1px solid #E1E8ED;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

.confidence-breakdown {
    display: flex;
    gap: 15px;
    margin-top: 20px;
}

.confidence-item {
    flex: 1;
    text-align: center;
    padding: 15px;
    border-radius: 10px;
    background: white;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    transition: transform 0.3s ease;
}

.confidence-item:hover {
    transform: translateY(-3px);
}

.confidence-value {
    font-size: 1.8rem;
    font-weight: 800;
    margin: 5px 0;
}

.confidence-label {
    font-size: 0.9rem;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.footer {
    text-align: center;
    padding: 25px;
    color: #657786;
    font-size: 0.95rem;
    background: rgba(255,255,255,0.7);
    border-radius: 15px;
    margin-top: 30px;
}

.quick-stats {
    display: flex;
    justify-content: center;
    gap: 30px;
    margin-top: 15px;
    font-size: 0.85rem;
}

.stat-item {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div style="text-align:center; margin-bottom: 40px;">
        <h1 style="color:#1DA1F2; margin-bottom: 15px; font-size: 3rem;">
            🐦 TweetScope
        </h1>
        <p style="color:#657786; font-size: 1.2rem; margin-bottom: 5px;">
            <b>Binary Sentiment Analysis</b> - Positive vs Negative
        </p>
        <p style="color:#94a3b8; font-size: 1rem;">
            Powered by Machine Learning | Real-time Classification
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Main Card
# -----------------------------
st.markdown("<div class='main-card'>", unsafe_allow_html=True)

# Header with refresh button
col1, col2 = st.columns([4, 1])
with col1:
    st.markdown(
        "<h2 style='color:#1e293b; margin-bottom: 5px;'>📊 Analyze Tweet Sentiment</h2>"
        "<p style='color:#64748b; margin-bottom: 25px;'>"
        "Enter a tweet below to classify it as Positive or Negative</p>",
        unsafe_allow_html=True
    )
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
    height=130,
    label_visibility="collapsed",
    key="tweet_input"
)

# Example tweets (Binary only - Positive/Negative)
st.markdown("<div class='example-buttons'>", unsafe_allow_html=True)
st.markdown("<p style='color:#64748b; margin-bottom: 10px; font-weight: 600;'>Try example tweets:</p>", unsafe_allow_html=True)

c1, c2 = st.columns(2)

# Updated examples for binary classification
positive_examples = [
    "Just had the best coffee ever! This new cafe is amazing! The barista was so friendly and the atmosphere is perfect. Highly recommend! #coffeelovers ☕️",
    "I'm absolutely loving the new update! The interface is sleek and everything works flawlessly. Great job team! 👏🎉",
    "Just finished an incredible workout! Feeling energized and ready to conquer the day. Fitness journey going strong! 💪 #fitnessmotivation"
]

negative_examples = [
    "Worst customer service ever. Waited 2 hours on hold just to be disconnected. Completely disappointed. Never buying again! #customerservicefail 😡",
    "The product arrived damaged and customer support is ignoring my emails. Very frustrating experience. Would not recommend to anyone. 📦❌",
    "Traffic is absolutely terrible today! Stuck for over an hour with no movement. This is ridiculous! 🚗💨"
]

with c1:
    if st.button("😊 Positive Examples", key="positive_example"):
        import random
        st.session_state.example = random.choice(positive_examples)
        st.experimental_rerun()

with c2:
    if st.button("😠 Negative Examples", key="negative_example"):
        import random
        st.session_state.example = random.choice(negative_examples)
        st.experimental_rerun()

st.markdown("</div>", unsafe_allow_html=True)

# Action buttons
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    analyze = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
with col2:
    if st.button("🗑️ Clear", type="secondary", use_container_width=True):
        st.session_state.example = ""
        st.session_state.last_analysis = None
        st.experimental_rerun()
with col3:
    if st.button("📋 Examples", type="secondary", use_container_width=True):
        st.session_state.example = "This is amazing! I love it so much! Best experience ever! 😍"
        st.experimental_rerun()

# -----------------------------
# Prediction Logic for Binary Classification
# -----------------------------
def analyze_sentiment(text):
    """Analyze sentiment for binary classification (Positive/Negative)"""
    tweet_vec = vectorizer.transform([text])
    prediction = model.predict(tweet_vec)[0]
    
    # Get confidence scores
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(tweet_vec)[0]
        
        # Assuming model returns [negative_prob, positive_prob] or [positive_prob, negative_prob]
        # Let's check the shape and adjust accordingly
        if len(proba) == 2:
            # Binary classification
            if prediction == 1:  # Positive
                confidence = proba[1] * 100
                confidence_scores = {
                    "Positive": proba[1] * 100,
                    "Negative": proba[0] * 100
                }
            else:  # Negative (prediction == 0)
                confidence = proba[0] * 100
                confidence_scores = {
                    "Positive": proba[1] * 100,
                    "Negative": proba[0] * 100
                }
        else:
            # Fallback if shape is different
            confidence = np.max(proba) * 100
            confidence_scores = {
                "Positive": proba[1] * 100 if len(proba) > 1 else 50,
                "Negative": proba[0] * 100 if len(proba) > 0 else 50
            }
    else:
        # If model doesn't have predict_proba
        confidence = 85.0
        confidence_scores = {
            "Positive": 85.0 if prediction == 1 else 15.0,
            "Negative": 85.0 if prediction == 0 else 15.0
        }
    
    # Map prediction to sentiment (0 = Negative, 1 = Positive)
    if prediction == 1:
        result = {
            "label": "POSITIVE",
            "emoji": "😊",
            "color": "var(--positive)",
            "card_class": "positive-card"
        }
    else:  # prediction == 0
        result = {
            "label": "NEGATIVE",
            "emoji": "😠",
            "color": "var(--negative)",
            "card_class": "negative-card"
        }
    
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
                # Add a small delay for better UX
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                result = analyze_sentiment(tweet)
                st.session_state.last_analysis = result
                
                # Clear progress bar
                progress_bar.empty()
        else:
            result = st.session_state.last_analysis
        
        # Display results with enhanced card
        st.markdown("---")
        
        # Result card with dynamic class
        st.markdown(f"<div class='result-card {result['card_class']}'>", unsafe_allow_html=True)
        
        # Sentiment display
        st.markdown("<div class='sentiment-display'>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='sentiment-emoji'>{result['emoji']}</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div class='sentiment-text' style='color:{result[\"color\"]};'>"
            f"{result['label']}</div>",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Confidence display
        st.markdown("<div class='confidence-container'>", unsafe_allow_html=True)
        
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="font-weight: 600; color:#475569;">Confidence Level</span>
                <span style="font-weight: 700; color:{result['color']}; font-size: 1.2rem;">
                    {result['confidence']:.1f}%
                </span>
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width:{result['confidence']}%; background:{result['color']};">
                    <div class="confidence-text">{result['confidence']:.1f}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Confidence breakdown
        st.markdown("<div class='confidence-breakdown'>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(
                f"""
                <div class='confidence-item' style='border-top: 4px solid var(--positive);'>
                    <div class='confidence-label'>Positive</div>
                    <div class='confidence-value' style='color:var(--positive);'>
                        {result['confidence_scores']['Positive']:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f"""
                <div class='confidence-item' style='border-top: 4px solid var(--negative);'>
                    <div class='confidence-label'>Negative</div>
                    <div class='confidence-value' style='color:var(--negative);'>
                        {result['confidence_scores']['Negative']:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("</div>", unsafe_allow_html=True)  # Close confidence-breakdown
        st.markdown("</div>", unsafe_allow_html=True)  # Close confidence-container
        
        # Original tweet display
        st.markdown("<div class='tweet-display'>", unsafe_allow_html=True)
        st.markdown("**📝 Original Tweet:**")
        st.markdown(f"> *{tweet}*")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Action buttons after analysis
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Analyze Another Tweet", type="secondary", use_container_width=True):
                st.session_state.example = ""
                st.session_state.last_analysis = None
                st.experimental_rerun()
        with col2:
            if st.button("📊 View Details", type="secondary", use_container_width=True):
                st.info("""
                **Classification Details:**
                - **Model Type:** Binary Classifier
                - **Output:** Positive (1) / Negative (0)
                - **Features:** TF-IDF Vectorization
                - **Confidence:** Probability score from model
                """)
        
        st.markdown("</div>", unsafe_allow_html=True)  # Close result-card

st.markdown("</div>", unsafe_allow_html=True)  # Close main-card

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    """
    <div class="footer">
        <div style="margin-bottom: 15px; font-size: 1.1rem;">
            Built with ❤️ by <b style="color:#1DA1F2;">Prabhat Yadav</b> | Powered by <b>Streamlit</b> & <b>Scikit-learn</b>
        </div>
        <div style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 20px;">
            TweetScope v2.0 | Binary Sentiment Analysis | Real-time Classification
        </div>
        <div class="quick-stats">
            <div class="stat-item">
                <span>📊</span>
                <span>Binary Classification</span>
            </div>
            <div class="stat-item">
                <span>⚡</span>
                <span>Real-time Analysis</span>
            </div>
            <div class="stat-item">
                <span>🔒</span>
                <span>Local Processing</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
