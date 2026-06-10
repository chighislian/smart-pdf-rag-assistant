# Project Architecture
                +----------------+
                | PDF Document   |
                +-------+--------+
                        |
                        v
              +------------------+
              | Text Extraction  |
              +--------+---------+
                       |
                       v
              +------------------+
              | Text Chunking    |
              +--------+---------+
                       |
                       v
              +------------------+
              | Embeddings Model |
              +--------+---------+
                       |
                       v
                +------------+
                | ChromaDB   |
                +------+-----+
                       |
                       v
User Question --> Similarity Search
                       |
                       v
                  Ollama LLM
                       |
                       v
                Final Answer




## Project workflow

Upload PDF
     ↓
Extract Text
     ↓
Split into Chunks
     ↓
Create Embeddings
     ↓
Store in ChromaDB/FAISS
     ↓
User asks question
     ↓
Find relevant chunks
     ↓
Send chunks + question to Ollama
     ↓
Generate answer


