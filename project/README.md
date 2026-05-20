# Smart Notes RAG Assistant

A beginner-friendly AI web app for uploading PDF or text documents, searching them semantically, and asking document-grounded questions through a conversational RAG chatbot.

The project is designed to be clean, modular, explainable, and resume-ready. It uses Streamlit for the UI, Sentence Transformers for embeddings, FAISS for vector search, and Ollama as the default local LLM provider.

## Features

- Upload PDF and TXT documents
- Extract readable text from uploaded files
- Split documents into overlapping chunks
- Generate embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Store vectors locally with FAISS
- Run cosine-similarity semantic search
- Ask document-grounded questions in a chat interface
- Show source chunks used for answers
- Preserve chat history during the session
- Support local Ollama models by default
- Optional OpenAI support through environment variables

## Tech Stack

| Layer | Technology |
| --- | --- |
| Frontend | Streamlit |
| Backend | Python |
| RAG framework | LangChain |
| Embeddings | Sentence Transformers |
| Vector database | FAISS |
| Local LLM | Ollama |
| Optional cloud LLM | OpenAI |
| PDF parsing | pypdf |

## Architecture

```text
Streamlit UI (app.py)
    |
    v
File Upload (PDF / TXT)
    |
    v
Text Extraction (utils/pdf_loader.py)
    |
    v
Text Chunking (utils/text_chunker.py)
    |
    v
Embedding Generation (utils/embedding_manager.py)
    |
    v
FAISS Vector Store (utils/vector_store.py)
    |
    +----------------------------+
    |                            |
    v                            v
Semantic Search             RAG Chatbot
(utils/retrieval.py)        (utils/rag_pipeline.py)
                                 |
                                 v
                         Ollama / Optional OpenAI
```

## RAG Flow

```text
Document Upload
-> Text Extraction
-> Text Chunking
-> Embedding Generation
-> FAISS Vector Storage
-> Similarity Retrieval
-> Context Injection
-> LLM Response Generation
-> Source Chunk Display
```

The app uses normalized embeddings with FAISS inner-product search. When vectors are normalized, inner product behaves like cosine similarity.

## Project Structure

```text
project/
|-- app.py
|-- requirements.txt
|-- README.md
|-- .env.example
|-- data/
|-- vectorstore/
|-- embeddings/
`-- utils/
    |-- pdf_loader.py
    |-- text_chunker.py
    |-- embedding_manager.py
    |-- vector_store.py
    |-- retrieval.py
    |-- rag_pipeline.py
    `-- ui_helpers.py
```

## Setup Instructions

### 1. Create a virtual environment

```powershell
cd project
python -m venv venv
.\venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
cd project
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Create your environment file

```powershell
Copy-Item .env.example .env
```

On macOS or Linux:

```bash
cp .env.example .env
```

Default configuration:

```text
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TOP_K_RESULTS=4
CHUNK_SIZE=900
CHUNK_OVERLAP=180
```

## Ollama Setup

Install Ollama from:

```text
https://ollama.com
```

Start the Ollama server:

```powershell
ollama serve
```

Pull a local model:

```powershell
ollama pull mistral
```

Alternative model:

```powershell
ollama pull llama3
```

If you use `llama3`, update `.env`:

```text
OLLAMA_MODEL=llama3
```

## Run the App

```powershell
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## How To Use

1. Upload one or more PDF or TXT files from the sidebar.
2. Choose the LLM provider and model settings.
3. Adjust chunk size and overlap if needed.
4. Click **Process documents**.
5. Use the **Semantic Search** tab to find relevant passages.
6. Use the **Chat** tab to ask questions about your documents.
7. Open **Latest Sources** to inspect the chunks used for the latest answer.

## Module Guide

| File | Responsibility |
| --- | --- |
| `app.py` | Main Streamlit app and user flow |
| `utils/pdf_loader.py` | PDF/TXT loading and text extraction |
| `utils/text_chunker.py` | Text cleaning and overlapping chunk creation |
| `utils/embedding_manager.py` | Sentence Transformer loading and embedding generation |
| `utils/vector_store.py` | FAISS index creation, saving, loading, and search |
| `utils/retrieval.py` | Semantic search wrapper for user queries |
| `utils/rag_pipeline.py` | RAG prompt construction and LLM answer generation |
| `utils/ui_helpers.py` | Reusable Streamlit display helpers |

## Validation And Testing

Run syntax checks:

```powershell
python -m py_compile app.py utils\pdf_loader.py utils\text_chunker.py utils\embedding_manager.py utils\vector_store.py utils\retrieval.py utils\rag_pipeline.py utils\ui_helpers.py
```

Test text chunking:

```powershell
python -c "from utils.text_chunker import split_text_into_chunks; chunks = split_text_into_chunks('abcdefghijklmnopqrstuvwxyz', 'sample.txt', 10, 3); print([chunk.text for chunk in chunks])"
```

Test prompt building:

```powershell
python -c "from utils.rag_pipeline import build_rag_prompt; print(build_rag_prompt('What is RAG?', 'RAG means retrieval augmented generation.')[:300])"
```

After installing dependencies, test embeddings:

```powershell
python -c "from utils.embedding_manager import generate_query_embedding; print(generate_query_embedding('What is RAG?').shape)"
```

## Error Handling

The app handles common beginner-project failure cases:

- Empty uploads
- Unsupported file types
- Corrupted or encrypted PDFs
- Empty extracted text
- Empty retrieval results
- Missing vector store
- Missing Ollama server
- Missing OpenAI API key when OpenAI is selected

## Screenshots

Add screenshots here after running the app:

```text
screenshots/
|-- upload_sidebar.png
|-- semantic_search.png
|-- rag_chat.png
`-- source_chunks.png
```

Suggested screenshot moments:

- Sidebar after uploading documents
- Successful document processing message
- Semantic search result expanders
- Chat answer with source chunks

## Resume Description

Built a Streamlit-based Smart Notes RAG Assistant that enables users to upload PDF/TXT documents, generate SentenceTransformer embeddings, store them in a FAISS vector database, and query them through a conversational LangChain + Ollama RAG pipeline with source-grounded answers.

## Interview Talking Points

- Why documents are split into overlapping chunks
- How normalized embeddings allow FAISS inner-product search to work as cosine similarity
- Why source chunks reduce hallucination risk
- How local Ollama support makes the app private and low-cost
- Why the project separates UI, extraction, chunking, embeddings, retrieval, and generation into different modules

## Future Improvements

- Add chat export to Markdown or JSON
- Add source citation highlighting inside answers
- Add per-document filtering
- Add persistent multi-user sessions
- Add a `tests/` folder with pytest coverage
- Add document deletion and vector store rebuilding
- Add support for DOCX and Markdown files
- Add advanced reranking for better retrieval quality

## Notes

This project intentionally avoids Docker, Kubernetes, complex agent frameworks, and unnecessary abstractions. The goal is to show a complete AI engineering workflow in a form that is easy to understand, run, explain, and extend.
