import streamlit as st
import joblib
import numpy as np

# Load model & vectorizer
model = joblib.load("trained_model.sav")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Page configuration
st.set_page_config(
    page_title="TweetScope | Sentiment Analysis",
    page_icon="🐦",
    layout="centered"
)

# Custom CSS for better UI
st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa;
        }
        .card {
            background-color: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        }
        .title {
            text-align: center;
            color: #1DA1F2;
        }
        .subtitle {
            text-align: center;
            color: gray;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='title'>🐦 TweetScope</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-powered Twitter Sentiment Analysis</p>", unsafe_allow_html=True)

st.write("")

# Card layout
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    tweet = st.text_area(
        "✍️ Enter a tweet",
        placeholder="Example: I really love this product!",
        height=150
    )

    analyze = st.button("🔍 Analyze Sentiment")

    if analyze:
        if tweet.strip() == "":
            st.warning("⚠️ Please enter a tweet.")
        else:
            tweet_vec = vectorizer.transform([tweet])
            prediction = model.predict(tweet_vec)[0]
            confidence = np.max(model.predict_proba(tweet_vec)) * 100

            st.markdown("---")

            if prediction == 1:
                st.success(f"😊 **Positive Sentiment**")
            elif prediction == 0:
                st.error(f"😠 **Negative Sentiment**")
            else:
                st.info(f"😐 **Neutral Sentiment**")

            st.write(f"**Confidence:** {confidence:.2f}%")

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.write("")
st.markdown("---")
st.markdown(
    "<center>Built with ❤️ by <b>Prabhat Yadav</b></center>",
    unsafe_allow_html=True
)
