import streamlit as st
import os
from ollama import chat
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from ingest import load_pdf, split_text

# Page Setup

st.set_page_config(
    page_title="PDF AI Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF AI Assistant")

# Session State


if "messages" not in st.session_state:
    st.session_state.messages = []

# Embeddings + Persistent DB

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

os.makedirs("vector_db", exist_ok=True)

db = Chroma(
    persist_directory="vector_db",
    embedding_function=embedding_model
)

# Sidebar

with st.sidebar:

    st.header("📚 PDF AI Assistant")

    st.write("✨ AI-Powered Document Intelligence")
    st.write("🚀 Smart Document Search")

    # Upload PDF (incremental indexing)

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:

        os.makedirs("uploads", exist_ok=True)

        file_path = os.path.join("uploads", uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success("PDF uploaded!")

        with st.spinner("Indexing document..."):

            text = load_pdf(file_path)
            chunks = split_text(text)

            docs = []

            for chunk in chunks:

                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={"source": uploaded_file.name}
                    )
                )

            # IMPORTANT: ADD instead of rebuild
            db.add_documents(docs)
            db.persist()

        st.success("Added to knowledge base!")

    # Show uploaded files

    pdf_files = []

    if os.path.exists("uploads"):

        pdf_files = [
            f for f in os.listdir("uploads")
            if f.endswith(".pdf")
        ]

    st.write(f"📄 Documents: {len(pdf_files)}")

    if pdf_files:

        st.write("### Uploaded Files")

        for file in pdf_files:

            st.write(f"📄 {file}")

    st.divider()

    # Clear Chat

    if st.button("🗑 Clear Chat"):

        st.session_state.messages = []
        st.rerun()

# Chat History

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input

question = st.chat_input("Ask a question about your documents...")

if question:

    # Save user message
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)


    # Retrieve context


    docs = db.similarity_search_with_score(question, k=6)

    docs = [
        doc for doc, score in docs
        if score < 1.2
    ]

    if not docs:
        docs = db.similarity_search(question, k=3)

    context = "\n\n".join(

        f"[Source chunk]\n{doc.page_content.strip()}"

        for doc in docs

    )

    # Prompt


    prompt = f"""
You are a PDF Question Answering AI Assistant.

Rules:
1. Use ONLY the provided context.
2. If the answer is not found, say:
   "The answer is not available in the uploaded documents."
3. Do not use outside knowledge.
4. Give clean and structured answers.

Context:
{context}

Question:
{question}

Answer:
"""

    
    # LLM Response


    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = chat(
                model="llama3.2:latest",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            answer = response["message"]["content"]

            st.markdown(answer)

    # Save assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )