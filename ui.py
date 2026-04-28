import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="AI Knowledge Assistant", layout="centered")
st.title("🤖 AI Knowledge Assistant")

# -------------------- LOAD DATABASE --------------------
@st.cache_resource
def load_db():
    loader = TextLoader("data.txt")
    docs = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

db = load_db()

# -------------------- LOAD MODEL (CLOUD SAFE FIX) --------------------
@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",        # ✅ fixed task
        model="distilgpt2"        # ✅ stable model
    )

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

            # 🔹 RAG retrieval
            results = db.similarity_search(query, k=2)
            context = " ".join([r.page_content for r in results])

            # 🔹 Prompt
            prompt = f"""
Answer the question based on the context below.

Context:
{context}

Question:
{query}

Answer:
"""

            # 🔹 Generate response
            response = generator(
                prompt,
                max_length=150,
                do_sample=True,
                temperature=0.7
            )

            answer = response[0]["generated_text"]

            # 🔹 CLEAN OUTPUT (IMPORTANT)
            answer = answer.replace(prompt, "").strip()

            # fallback
            if len(answer) < 5:
                answer = "Sorry, I couldn't find a clear answer in the data."

            st.markdown(answer)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})