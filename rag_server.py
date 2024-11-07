from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from chromadb.client import ChromaClient
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from docx import Document
from pydantic import BaseModel
from typing import List

app = FastAPI()

client = ChromaClient(persist_directory="chromadb_data")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

async def read_file(file: UploadFile) -> str:
    content = ""
    if file.filename.endswith(".pdf"):
        reader = PdfReader(file.file)
        content = " ".join([page.extract_text() for page in reader.pages])
    elif file.filename.endswith((".doc", ".docx")):
        doc = Document(file.file)
        content = " ".join([para.text for para in doc.paragraphs])
    elif file.filename.endswith(".txt"):
        content = (await file.read()).decode("utf-8")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format.")
    return content

@app.post("/ingest/")
async def ingest_document(file: UploadFile):
    try:
        content = await read_file(file)
        embedding = model.encode(content)
        client.add_documents([{"content": content, "embedding": embedding}])
        return JSONResponse(content={"status": "Document ingested successfully."}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {str(e)}")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/query/")
async def query_documents(request: QueryRequest):
    try:
        embedding = model.encode(request.query)
        results = client.query_embeddings([embedding], top_k=request.top_k)
        response = [{"document": res["content"], "score": res["score"]} for res in results]
        return JSONResponse(content=response, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query documents: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
