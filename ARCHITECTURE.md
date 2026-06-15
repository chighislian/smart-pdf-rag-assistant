# PDF QA Chatbot (RAG) Architecture

This project is a Retrieval-Augmented Generation (RAG) system that lets users upload PDFs and ask questions about them using a local LLM (Ollama).

## Project Architecture:


> **Stack:** Streamlit · LangChain · HuggingFace · ChromaDB · Ollama


## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI                             │
│              PDF upload  ·  Question input  ·  Chat display     │
└───────────────────────┬─────────────────────────────────────────┘
                        │
          ┌─────────────┴─────────────┐
          │                           │
          ▼                           ▼
┌─────────────────────┐   ┌─────────────────────┐
│  DOCUMENT INGESTION │   │   QUERY PROCESSING  │
│                     │   │                     │
│  • Load PDF         │   │  • Receive question │
│  • Chunk text       │   │  • Embed query      │
│  • Clean content    │   │                     │
└──────────┬──────────┘   └──────────┬──────────┘
           │                         │
           └───────────┬─────────────┘
                       │
                       ▼
          ┌────────────────────────────┐
          │      EMBEDDING MODEL       │
          │   HuggingFaceEmbeddings    │
          │  (sentence-transformers)   │
          └────────────┬───────────────┘
                       │
                       ▼
          ┌────────────────────────────┐
          │       VECTOR STORE         │
          │         ChromaDB           │
          │  Persists chunk embeddings │
          └────────────┬───────────────┘
                       │  top-k similarity search
                       ▼
          ┌────────────────────────────┐
          │    PROMPT CONSTRUCTION     │
          │  Retrieved chunks + Query  │
          │   (LangChain template)     │
          └────────────┬───────────────┘
                       │
                       ▼
          ┌────────────────────────────┐
          │      LANGUAGE MODEL        │
          │    Ollama  (local)         │
          │   Llama 3  /  Mistral      │
          └────────────┬───────────────┘
                       │
                       ▼
          ┌────────────────────────────┐
          │         RESPONSE           │
          │  Streamed back to UI       │
          └────────────────────────────┘
```

-----

## Component Breakdown

|Layer            |Component                 |Role                                     |
|-----------------|--------------------------|-----------------------------------------|
|**Interface**    |Streamlit (`app.py`)      |User-facing UI — upload, chat, display   |
|**Ingestion**    |LangChain PDF Loader      |Load and split PDF into text chunks      |
|**Embedding**    |HuggingFaceEmbeddings     |Encode chunks and queries into vectors   |
|**Storage**      |ChromaDB                  |Persist and index chunk embeddings       |
|**Retrieval**    |Similarity Search         |Return top-k relevant chunks for a query |
|**Orchestration**|LangChain Chain           |Assemble context + question into a prompt|
|**Generation**   |Ollama (Llama 3 / Mistral)|Generate grounded answer locally         |

-----

## Data Flow

1. **Upload** — User uploads a PDF via Streamlit UI
1. **Chunk** — PDF is split into overlapping text chunks
1. **Embed** — Each chunk is encoded into a dense vector
1. **Index** — Vectors are stored in ChromaDB
1. **Query** — User question is embedded using the same model
1. **Retrieve** — Top-k most similar chunks are fetched from ChromaDB
1. **Augment** — Retrieved context is injected into the prompt template
1. **Generate** — Ollama runs the LLM locally and produces an answer
1. **Display** — Response is streamed back into the Streamlit chat UI