import streamlit as st
import joblib
import numpy as np
from streamlit_extras.colored_header import colored_header
from streamlit_extras.stylable_container import stylable_container

# Load model & vectorizer
model = joblib.load("trained_model.sav")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Page configuration
st.set_page_config(
    page_title="TweetScope | Sentiment Analysis",
    page_icon="🐦",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
    <style>
        :root {
            --primary: #1DA1F2;
            --positive: #00C853;
            --negative: #FF5252;
            --neutral: #FFC107;
            --bg: #f0f2f6;
        }
        
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e7eb 100%);
        }
        
        .main-card {
            background: white;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.08);
            padding: 2rem;
            margin-top: 1rem;
            transition: all 0.3s ease;
        }
        
        .tweet-input {
            border: 2px solid #e6ecf0 !important;
            border-radius: 12px !important;
            padding: 15px !important;
            font-size: 16px !important;
            transition: border 0.3s ease !important;
        }
        
        .tweet-input:focus {
            border-color: var(--primary) !important;
            outline: none !important;
            box-shadow: 0 0 0 3px rgba(29, 161, 242, 0.2) !important;
        }
        
        .stButton>button {
            background: var(--primary) !important;
            color: white !important;
            border-radius: 50px !important;
            padding: 12px 24px !important;
            font-weight: 600 !important;
            font-size: 16px !important;
            transition: all 0.3s ease !important;
            border: none !important;
            width: 100%;
        }
        
        .stButton>button:hover {
            background: #0d8bd9 !important;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(29, 161, 242, 0.25);
        }
        
        .result-card {
            padding: 1.5rem;
            border-radius: 16px;
            margin-top: 1.5rem;
            text-align: center;
        }
        
        .confidence-bar {
            height: 12px;
            border-radius: 10px;
            background: #e0e0e0;
            margin: 20px 0;
            overflow: hidden;
        }
        
        .confidence-fill {
            height: 100%;
            border-radius: 10px;
        }
        
        .example-tweet {
            padding: 10px 15px;
            background: #f5f8fa;
            border-radius: 12px;
            margin: 8px 0;
            cursor: pointer;
            transition: all 0.2s ease;
            border-left: 3px solid var(--primary);
        }
        
        .example-tweet:hover {
            background: #e8f5fe;
            transform: translateX(5px);
        }
        
        footer {
            text-align: center;
            margin-top: 2rem;
            color: #657786;
            font-size: 14px;
        }
    </style>
""", unsafe_allow_html=True)

# Header with gradient
colored_header(
    label="",
    description="",
    color_name="blue-70",
    anchor="tweet-scope",
    label_visibility="collapsed"
)

st.markdown("<div style='text-align:center; margin-top:-50px;'>", unsafe_allow_html=True)
st.markdown("<h1 style='color:#1DA1F2; font-size:2.8rem;'>🐦 TweetScope</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#657786; font-size:1.1rem;'>AI-powered Twitter Sentiment Analysis</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Main card container
with stylable_container(
    key="main-card",
    css_styles="""
        {
            background: white;
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        }
    """
):
    # Input section
    st.subheader("📝 Analyze Tweet Sentiment")
    st.caption("Enter text below to analyze sentiment")
    
    tweet = st.text_area(
        "Enter tweet text:",
        placeholder="What's happening? Share your thoughts...",
        height=150,
        label_visibility="collapsed"
    )
    
    # Example tweets
    st.caption("Try these examples:")
    examples = st.columns(3)
    with examples[0]:
        if st.button("😊 Positive", use_container_width=True):
            st.session_state.example = "Just had the best coffee ever! This new cafe is amazing! #coffeelover"
    with examples[1]:
        if st.button("😠 Negative", use_container_width=True):
            st.session_state.example = "Worst customer service ever! Waited 2 hours and they got my order wrong."
    with examples[2]:
        if st.button("😐 Neutral", use_container_width=True):
            st.session_state.example = "The weather forecast predicts rain tomorrow in Seattle."
    
    # Set example if selected
    if 'example' in st.session_state:
        tweet = st.session_state.example
    
    # Analyze button
    analyze = st.button("🔍 Analyze Sentiment", type="primary")
    
    # Results section
    if analyze:
        if not tweet.strip():
            st.warning("⚠️ Please enter a tweet to analyze")
        else:
            with st.spinner("Analyzing sentiment..."):
                tweet_vec = vectorizer.transform([tweet])
                prediction = model.predict(tweet_vec)[0]
                confidence = np.max(model.predict_proba(tweet_vec)) * 100
                
                # Display results
                st.markdown("---")
                
                # Result card with color coding
                if prediction == 1:
                    result_color = "var(--positive)"
                    emoji = "😊"
                    sentiment = "Positive"
                elif prediction == 0:
                    result_color = "var(--negative)"
                    emoji = "😠"
                    sentiment = "Negative"
                else:
                    result_color = "var(--neutral)"
                    emoji = "😐"
                    sentiment = "Neutral"
                
                # Animated result display
                with stylable_container(
                    key="result-card",
                    css_styles=f"""
                        {{
                            background: {result_color}10;
                            border: 1px solid {result_color}30;
                            border-radius: 16px;
                            padding: 1.5rem;
                            text-align: center;
                            animation: fadeIn 0.5s ease-in;
                        }}
                        
                        @keyframes fadeIn {{
                            0% {{ opacity: 0; transform: translateY(10px); }}
                            100% {{ opacity: 1; transform: translateY(0); }}
                        }}
                    """
                ):
                    st.markdown(f"<h2 style='margin-bottom:10px;'>{emoji} {sentiment} Sentiment</h2>", unsafe_allow_html=True)
                    
                    # Confidence bar
                    st.markdown(f"<div style='color:#657786; margin-bottom:8px;'>Confidence: {confidence:.1f}%</div>", unsafe_allow_html=True)
                    st.markdown(
                        f"""
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width:{confidence}%; background:{result_color};"></div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Display tweet preview
                    st.markdown("---")
                    st.caption("Your Tweet:")
                    st.info(tweet)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center; padding:20px; color:#657786;'>"
    "Built with ❤️ by <b>Prabhat Yadav</b> | "
    "Powered by Streamlit 🤖"
    "</div>",
    unsafe_allow_html=True
)
