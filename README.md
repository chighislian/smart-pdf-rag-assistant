# 📄 Smart PDF RAG Chatbot

An AI-powered Retrieval-Augmented Generation (RAG) assistant that allows users to upload PDF documents and have real-time, context-grounded conversations with them. The system operates entirely locally using **Ollama** for LLM inference, **HuggingFace** for embeddings, and **ChromaDB** for vector storage.

---

## 🚀 Key Features

* **Multiple Interfaces**: 
  * **Streamlit Web Application (`app.py`)**: A premium web-based chat interface supporting file uploads, live streaming responses, and references to source documents.
  * **Interactive CLI (`chatbot.py`)**: A fast, terminal-based chatbot for quick document querying.
* **Local Processing & Privacy**: No document data or queries leave your local machine.
* **Document Ingestion (`ingest.py`)**: Seamless PDF text extraction with configurable text splitting and chunk overlap parameters.
* **Persistent Vector Storage**: Powered by ChromaDB (`vector_db/`), allowing indexing to persist across application restarts.
* **Real-time Answer Streaming**: Streamlit chat uses response streaming to display the model's response token-by-token as it's generated.
* **Source Attribution**: Highlights the specific documents used to form the answer.

---

## 🛠️ Tech Stack

* **Frontend**: Streamlit
* **Orchestration**: LangChain (Community & Core)
* **Embeddings**: HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
* **Vector Database**: ChromaDB
* **Local Inference**: Ollama (`llama3.2:latest`)
* **PDF Parser**: PyPDF

---

## 📐 System Architecture

For a visual walkthrough of the RAG pipeline:
* See [ARCHITECTURE.md](file:///c:/Project/PDF_RAG_ChatBot/ARCHITECTURE.md) for a detailed diagram and component breakdown.
* Open [index.html](file:///c:/Project/PDF_RAG_ChatBot/index.html) in your browser to view a rich interactive block diagram of the system layers.

### Ingestion & Query Pipeline:
1. **Upload / Load**: PDFs are read and processed page-by-page using `pypdf`.
2. **Chunking**: Text is split into overlapping chunks (e.g. 1000 characters with 200 overlap) to preserve context boundaries.
3. **Embedding**: Chunks are mapped into 384-dimensional dense vector spaces using HuggingFace's `all-MiniLM-L6-v2` transformer.
4. **Storage**: Vector representations and raw texts are indexed into a local `ChromaDB` directory.
5. **Retrieval**: User queries are embedded, and a similarity search retrieves the top-k relevant text chunks.
6. **Generation**: Retrieved context is injected into a strict system prompt and sent to Ollama (`llama3.2`) to generate a grounded final answer.

---

## 📂 Project Structure

* [app.py](file:///c:/Project/PDF_RAG_ChatBot/app.py): The main Streamlit web application.
* [chatbot.py](file:///c:/Project/PDF_RAG_ChatBot/chatbot.py): Terminal-based interactive CLI chatbot.
* [ingest.py](file:///c:/Project/PDF_RAG_ChatBot/ingest.py): Extractor and chunking utility using `pypdf`.
* [build_index.py](file:///c:/Project/PDF_RAG_ChatBot/build_index.py): Ingestion script that scans the `uploads` directory and rebuilds the vector database.
* [create_db.py](file:///c:/Project/PDF_RAG_ChatBot/create_db.py): Simple database initialization script for default sample files.
* [rag.py](file:///c:/Project/PDF_RAG_ChatBot/rag.py): Developer utility script to test similarity search and document retrieval in isolation.
* [requirements.txt](file:///c:/Project/PDF_RAG_ChatBot/requirements.txt): Pinned package dependencies.
* [index.html](file:///c:/Project/PDF_RAG_ChatBot/index.html): HTML-based interactive visual architecture map.
* [ARCHITECTURE.md](file:///c:/Project/PDF_RAG_ChatBot/ARCHITECTURE.md): Diagram representation of data flows.

---

## 🚦 Getting Started

### Prerequisites

1. **Python**: Ensure you have Python 3.8 or newer installed.
2. **Ollama**: Download and install Ollama from [ollama.com](https://ollama.com).
3. **Pull Model**: Once Ollama is running, pull the Llama 3.2 model:
   ```bash
   ollama pull llama3.2
   ```

### Installation

1. Navigate to the project root:
   ```bash
   cd c:\Project\PDF_RAG_ChatBot
   ```
2. Create and activate a Python virtual environment:
   ```powershell
   # Create virtual environment
   python -m venv venv

   # Activate virtual environment (Windows PowerShell)
   .\venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Running the Application

### 1. Launch the Streamlit Web UI (Recommended)
Start the Streamlit application:
```bash
streamlit run app.py
```
* A new browser tab should open automatically at `http://localhost:8501`.
* Use the sidebar to upload a PDF. Once indexed, you can immediately begin asking questions in the chat!

### 2. Launch the CLI Chatbot
Start the terminal interactive bot:
```bash
python chatbot.py
```
* Type your question and hit Enter to receive responses. Type `exit` to close.

### 3. Rebuild / Populate the Index Manually
If you drop new PDF files directly into the `uploads` folder and want to build the database outside the Streamlit app:
```bash
python build_index.py
```
This will remove any existing database, re-index all PDFs, and initialize a fresh store.
