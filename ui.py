import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="AI Knowledge Assistant", layout="centered")
st.title("🤖 AI Knowledge Assistant")

# -------------------- MODEL SELECTOR --------------------
model_choice = st.selectbox(
    "Choose model:",
    ["distilgpt2"]   # simplified for cloud
)

# -------------------- LOAD DATABASE --------------------
@st.cache_resource
def load_db():
    loader = TextLoader("data.txt")
    docs = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

db = load_db()

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="distilgpt2")

generator = load_model()

# -------------------- CHAT HISTORY --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------- USER INPUT --------------------
query = st.chat_input("Ask me anything...")

if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking... ⏳"):

            # 🔹 RAG retrieval (UNCHANGED)
            results = db.similarity_search(query, k=2)
            context = " ".join([r.page_content for r in results])

            prompt = f"""
You are a helpful AI assistant.

Use the context below to answer clearly and concisely.

Context:
{context}

Question:
{query}

Answer:
"""

            # 🔹 NEW MODEL (REPLACES OLLAMA)
            response = generator(
                prompt,
                max_length=200,
                num_return_sequences=1
            )

            answer = response[0]["generated_text"]

            # Clean output (optional)
            answer = answer.replace(prompt, "").strip()

            st.markdown(answer)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})