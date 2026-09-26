import re
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from faq_data import faq_questions, faq_answers

# ---------- NLP preprocessing ----------
STOP_WORDS = set(stopwords.words("english"))

def preprocess(text):
    """Clean and tokenize text for FAQ matching."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in STOP_WORDS]
    return " ".join(tokens)

# Preprocess the FAQ questions once.
processed_questions = [preprocess(q) for q in faq_questions]

# Convert FAQ questions into TF-IDF vectors.
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
faq_vectors = vectorizer.fit_transform(processed_questions)

def get_answer(user_question, threshold=0.20):
    """Return the best FAQ answer or a fallback response."""
    processed_query = preprocess(user_question)

    if not processed_query.strip():
        return "Please enter a question.", 0.0, None

    user_vector = vectorizer.transform([processed_query])
    similarities = cosine_similarity(user_vector, faq_vectors)[0]

    best_match = similarities.argmax()
    best_score = float(similarities[best_match])

    # IMPORTANT: reject weak matches when the score is BELOW the threshold.
    if best_score < threshold:
        return (
            "Sorry, I could not find a reliable answer to that question. "
            "Please try asking in a different way or contact customer support.",
            best_score,
            None,
        )

    return faq_answers[best_match], best_score, faq_questions[best_match]


# ---------- Streamlit UI ----------
st.set_page_config(
    page_title="AI FAQ Support Chatbot",
    page_icon="💬",
    layout="centered",
)

st.title("💬 AI FAQ Support Chatbot")
st.caption("An NLP-based FAQ assistant using TF-IDF and cosine similarity.")
st.write(
    "Ask a question about orders, delivery, returns, payments, or customer support."
)

user_question = st.text_input(
    "Enter your question:",
    placeholder="Example: How can I track my order?",
)

if st.button("Ask", type="primary"):
    if user_question.strip():
        answer, score, matched_question = get_answer(user_question)

        st.success(answer)

        if matched_question:
            with st.expander("View matching FAQ"):
                st.write(f"**Matched question:** {matched_question}")
                st.write(f"**Similarity score:** {score:.2f}")
    else:
        st.warning("Please enter a question.")

st.divider()
st.caption("Project: FAQ Chatbot | NLP: NLTK + TF-IDF | Matching: Cosine Similarity")