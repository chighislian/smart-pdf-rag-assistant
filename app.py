import streamlit as st

from ollama import chat

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma

st.set_page_config(page_title="PDF AI Assistant", page_icon="📄")

st.title("📄 PDF AI Assistant")

# Load embeddings + DB

embedding_model = HuggingFaceEmbeddings(

    model_name="sentence-transformers/all-MiniLM-L6-v2"

)

db = Chroma(

    persist_directory="vector_db",

    embedding_function=embedding_model

)

question = st.text_input("Ask a question")

if st.button("Ask") and question:

    docs = db.similarity_search(question, k=3)

    context = "\n\n".join([d.page_content for d in docs])

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

Answer clearly and concisely.

"""

    response = chat(

        model="llama3.2:latest",

        messages=[{"role": "user", "content": prompt}]

    )

    st.subheader("Answer")

    st.write(response["message"]["content"])