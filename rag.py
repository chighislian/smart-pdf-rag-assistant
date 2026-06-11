from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma

embedding_model = HuggingFaceEmbeddings(

    model_name="sentence-transformers/all-MiniLM-L6-v2"

)

db = Chroma(

    persist_directory="vector_db",

    embedding_function=embedding_model

)

query = "What is DevOps?"

results = db.similarity_search(query, k=3)

for i, doc in enumerate(results, start=1):

    print(f"\n===== Result {i} =====")

    print(doc.page_content[:500])