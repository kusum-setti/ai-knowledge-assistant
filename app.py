from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import ollama

# Load data
loader = TextLoader("data.txt")
docs = loader.load()

# Create embeddings
embeddings = HuggingFaceEmbeddings()

# Create vector DB
db = FAISS.from_documents(docs, embeddings)

print("🤖 Smart RAG Chatbot Ready! Type 'exit' to quit.\n")

while True:
    query = input("You: ")

    if query.lower() == "exit":
        print("Goodbye 👋")
        break

    # Retrieve relevant docs
    results = db.similarity_search(query)
    context = " ".join([r.page_content for r in results])

    # Create prompt
    prompt = f"""
You are a helpful AI assistant.

Use the context below to answer the question clearly.

Context:
{context}

Question:
{query}

Answer:
"""

    # Call Ollama
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    print("Bot:", response["message"]["content"])
    print("-" * 50)