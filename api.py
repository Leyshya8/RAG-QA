from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil, os
from retriever import load_qa_chain, ask
from ingest import ingest_documents

app = FastAPI(title="RAG Document Q&A API")

# Global variables for chain and retriever
chain = None
retriever = None

@app.on_event("startup")
def startup():
    global chain, retriever
    if os.path.exists("./faiss_index"):
        chain, retriever = load_qa_chain()

class QuestionRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF and re-index."""
    global chain, retriever

    os.makedirs("./docs", exist_ok=True)
    dest = f"./docs/{file.filename}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    ingest_documents()

    # Reload both chain and retriever
    chain, retriever = load_qa_chain()

    return {"message": f"'{file.filename}' indexed successfully"}

@app.post("/ask")
def ask_question(req: QuestionRequest):
    """Ask a question against the indexed documents."""
    if chain is None:
        raise HTTPException(status_code=400, detail="No documents indexed yet. Upload a PDF first.")
    return ask(req.question, chain, retriever)

@app.get("/health")
def health():
    return {"status": "ok", "indexed": chain is not None}