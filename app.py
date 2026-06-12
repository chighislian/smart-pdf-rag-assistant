import streamlit as st

from ollama import chat

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma

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

# Sidebar

with st.sidebar:

    st.header("PDF AI Assistant")

    st.write("🤖 Model: Llama 3.2")

    st.write("📚 Vector DB: ChromaDB")

    st.write("🔍 Embedding: all-MiniLM-L6-v2")

    st.divider()

    if st.button("🗑 Clear Chat"):

        st.session_state.messages = []

        st.rerun()

# Load Vector Database

embedding_model = HuggingFaceEmbeddings(

    model_name="sentence-transformers/all-MiniLM-L6-v2"

)

db = Chroma(

    persist_directory="vector_db",

    embedding_function=embedding_model

)

# Display Chat History

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# User Input


question = st.chat_input(

    "Ask a question about your documents..."

)

if question:

    # Show User Message

    st.session_state.messages.append(

        {

            "role": "user",

            "content": question

        }

    )

    with st.chat_message("user"):

        st.markdown(question)


    # Retrieve Context


    docs = db.similarity_search(

        question,

        k=5

    )

    context = "\n\n".join(

        [doc.page_content for doc in docs]

    )

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

    # Save Assistant Response

    st.session_state.messages.append(

        {

            "role": "assistant",

            "content": answer

        }

    )