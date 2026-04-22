import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def ingest_documents(docs_path: str = "./docs", index_path: str = "./faiss_index"):
    """Load documents, chunk them, embed, and save vector store."""

    # 1. Load documents (supports PDF; add more loaders as needed)
    print("📄 Loading documents...")
    loader = DirectoryLoader(docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"   Loaded {len(documents)} pages")

    # 2. Split into chunks
    # chunk_size: tokens per chunk | chunk_overlap: overlap to preserve context
    print("✂️  Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"   Created {len(chunks)} chunks")

    # 3. Create embeddings and store in FAISS
    print("🔢 Creating embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vector_store = FAISS.from_documents(chunks, embeddings)

    # 4. Save index to disk
    vector_store.save_local(index_path)
    print(f"✅ Index saved to '{index_path}'")

if __name__ == "__main__":
    os.makedirs("./docs", exist_ok=True)
    ingest_documents()