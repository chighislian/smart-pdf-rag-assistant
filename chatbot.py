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
You are a PDF Question Answering AI Assistant.
Use ONLY the provided context.
Rules:
1. Answer only from the context.
2. If the answer is not found, say:
   "The answer is not available in the uploaded document."
3. Do not use outside knowledge.
4. Be concise and accurate.

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