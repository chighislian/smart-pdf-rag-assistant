import os

import shutil

from ingest import load_pdf, split_text

from langchain_core.documents import Document

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma

UPLOAD_DIR = "uploads"

embedding_model = HuggingFaceEmbeddings(

    model_name="sentence-transformers/all-MiniLM-L6-v2"

)

all_documents = []

for file in os.listdir(UPLOAD_DIR):

    if file.endswith(".pdf"):

        path = os.path.join(UPLOAD_DIR, file)

        text = load_pdf(path)

        chunks = split_text(text)

        for i, chunk in enumerate(chunks):

            all_documents.append(

                Document(

                    page_content=chunk,

                    metadata={"source": file, "chunk": i}

                )

            )

# delete old db ONLY during indexing

if os.path.exists("vector_db"):

    shutil.rmtree("vector_db")

db = Chroma.from_documents(

    documents=all_documents,

    embedding=embedding_model,

    persist_directory="vector_db"

)

print("Index built successfully")