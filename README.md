# FastAPI RAG Server

This project implements a lightweight Retrieval-Augmented Generation (RAG) server using FastAPI, ChromaDB, and SentenceTransformers.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Kaushalsingh79/RAGserver.git
   ```

2. Activate the virtual environment:
   ```bash
   source rag_server/venv/bin/activate
   ```

3. Run the server:
   ```bash
   uvicorn rag_server:app --host 0.0.0.0 --port 8000 --workers 4
   ```

## Endpoints

- **POST /ingest/**: Upload a document (PDF, DOC, DOCX, TXT) for ingestion.
- **POST /query/**: Query the ingested documents with semantic search.

## License

Licensed under MIT.
