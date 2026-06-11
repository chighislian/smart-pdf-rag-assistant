from ingest import load_pdf, split_text
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


pdf_path = r"sample_pdf/DevOps_Material.pdf"

text = load_pdf(pdf_path)

chunks = split_text(text)

documents = [
    Document(page_content=chunk)
    for chunk in chunks
]

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory="vector_db"
)

print("Vector database created successfully!")