from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from ollama import chat

# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector database
db = Chroma(
    persist_directory="vector_db",
    embedding_function=embedding_model
)

while True:
    question = input("\nAsk a question (or type exit): ")

    if question.lower() == "exit":
        break

    # Retrieve relevant chunks
    docs = db.similarity_search(question, k=3)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are a precise AI assistant.

Rules:
- Do NOT repeat information
- Do NOT duplicate sentences
- Give a clean, structured answer
- Use ONLY the context below
Context:
{context}

Question:
{question}

Answer:
"""

    response = chat(
        model="llama3.2:latest",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print("\nAnswer:")
    print(response["message"]["content"])