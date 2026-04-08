# app.py

import streamlit as st
from dotenv import load_dotenv
from src.embeddings import get_embeddings
from src.vectorstore import load_vectorstore
from src.retriever import get_retriever
from src.prompt import get_prompt
from src.chain import get_chain
from src.guardrail import is_relevant_query, get_answer_with_sources
from config import CHROMA_DIR

load_dotenv()

# ── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="AI Learning Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Learning Assistant")
col1, col2 = st.columns([3, 1])
with col1:
    st.caption("Your personal Machine Learning and Deep Learning assistant")
with col2:
    if st.button("🗑️ Clear History"):
        st.session_state.messages = []
        # Reload chain to reset memory
        st.cache_resource.clear()
        st.rerun()


# ── Load Model Once Using Session State ──────────────────
@st.cache_resource
def load_tutor():
    embeddings = get_embeddings()
    vectorstore = load_vectorstore(embeddings, CHROMA_DIR)
    retriever = get_retriever(vectorstore, embeddings)
    prompt = get_prompt()
    chain = get_chain(retriever, prompt)
    return chain


chain = load_tutor()

# ── Chat History in Session State ────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Display Chat History ──────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Chat Input ────────────────────────────────────────────
from src.cleaner import preprocess_query
if user_input := st.chat_input("Ask me anything about ML/DL..."):

    # Show original message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Preprocess query
    cleaned_input = preprocess_query(user_input)

    # Show spelling correction notice if query changed
    if cleaned_input != user_input.lower().strip():
        st.info(f"🔍 Interpreted as: **{cleaned_input}**")

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    if not is_relevant_query(cleaned_input):
        response = "I am an ML/DL tutor. Please ask questions \
                    related to Machine Learning or Deep Learning."
    else:
        with st.spinner("Thinking..."):
            response = get_answer_with_sources(chain, cleaned_input)

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })