# =============================================
# Day 23 - RAG API with FastAPI
# Author: Abdelrhman
# Date: June 2026
# =============================================
# Wraps the RAG system (Days 15-20) in real endpoints:
# POST /upload - load, split, embed, store a document
# POST /ask    - search, prompt, get answer with sources
# =============================================

import os
import shutil
from fastapi import FastAPI, UploadFile, HTTPException, Header, Depends
from dotenv import load_dotenv

import google.genai as genai
from google.genai import types

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

from models import QuestionRequest, AnswerResponse, SourceInfo, UploadResponse

load_dotenv()

# =============================================
# AUTHENTICATION
# =============================================

API_SECRET_KEY = os.getenv("RAG_API_SECRET_KEY")


def verify_api_key(x_api_key: str = Header(...)):
    """
    Dependency function that checks the X-API-Key header
    on every request that uses it.

    Header(...) means: this value MUST come from an HTTP
    header named "X-API-Key" (FastAPI automatically converts
    x_api_key → X-API-Key). The "..." means it's REQUIRED —
    if missing, FastAPI rejects the request automatically
    before this function even runs.
    """
    if x_api_key != API_SECRET_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )
    return x_api_key

# Update the FastAPI app initialization:
app = FastAPI(
    title="RAG Document Q&A API",
    description="""
A Retrieval-Augmented Generation (RAG) API for document Q&A.

Upload PDF or text documents, then ask questions about their content.
Answers are grounded ONLY in the uploaded documents — if the answer
isn't found, the API will say so honestly instead of guessing.

**Authentication:** All endpoints except `/health` require an
`X-API-Key` header.
    """,
    version="1.0.0",
    contact={
        "name": "Abdelrhman",
        "url": "https://github.com/abdelrhman-builds"
    }
)


# =============================================
# GLOBAL STATE (simple version for today)
# =============================================
# In a real production app this would be a database.
# For today, we keep it simple: one shared vector store
# in memory while the server runs.

vectorstore = None
embeddings_model = None
gemini_client = None


@app.on_event("startup")
def load_models():
    """
    Runs ONCE when the API server starts up.
    Loads the embeddings model and Gemini client a single
    time, so every request reuses them instead of reloading.

    This is the FastAPI equivalent of Streamlit's
    @st.cache_resource from Day 19/20.
    """
    global embeddings_model, gemini_client

    print("Loading embeddings model...")
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Initializing Gemini client...")
    api_key = os.getenv("GEMINI_API_KEY")
    gemini_client = genai.Client(api_key=api_key)

    print("✅ Models loaded - API ready!")


# =============================================
# ENDPOINT 1: Upload Document
# =============================================

@app.post(
    "/upload",
    response_model=UploadResponse,
    tags=["Documents"],
    summary="Upload a document for Q&A",
    description="""
Uploads a PDF or TXT file, processes it through the RAG pipeline
(load → split → embed → store), and makes it available for
questions via the /ask endpoint.

**Supported formats:** PDF, TXT
**Max recommended size:** keep files reasonably small for fast processing
    """
)
def upload_document(file: UploadFile, api_key: str = Depends(verify_api_key)):
    """
    Depends(verify_api_key) tells FastAPI:
    "before running this function, run verify_api_key()
    first. If it raises an error, stop here and return
    that error instead."

    This is FastAPI's "dependency injection" pattern —
    reusable logic (auth, in this case) that multiple
    endpoints can share without copy-pasting the check
    into every single function.
    """
    global vectorstore

    # Validate file type (same security thinking as Day 20)
    if not (file.filename.endswith(".pdf") or file.filename.endswith(".txt")):
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported"
        )

    # Save temporarily to process it
    temp_path = f"./temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Load document (Day 15 logic)
        if file.filename.endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
        else:
            with open(temp_path, "r", encoding="utf-8") as f:
                text = f.read()
            documents = [Document(page_content=text, metadata={})]

        for doc in documents:
            doc.metadata["source"] = file.filename

        # Split into chunks (Day 15 logic)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i

        # Embed + Store (Day 16 logic)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory="./api_chroma_db"
        )

        return UploadResponse(
            filename=file.filename,
            chunks_created=len(chunks),
            status="success"
        )

    finally:
        # Always clean up, even if something failed
        if os.path.exists(temp_path):
            os.remove(temp_path)


# =============================================
# ENDPOINT 2: Ask a Question
# =============================================

@app.post(
    "/ask",
    response_model=AnswerResponse,
    tags=["Q&A"],
    summary="Ask a question about uploaded documents",
    description="""
Searches the uploaded document(s) for relevant content, then asks
Gemini to answer using ONLY that content.

If the answer isn't found in the documents, the response will
honestly say so rather than guessing from general knowledge.
    """
)
async def ask_question(request: QuestionRequest, api_key: str = Depends(verify_api_key)):
    """
    Now an ASYNC endpoint. The vector search (similarity_search)
    stays synchronous because Chroma's local search is FAST —
    it doesn't involve network waiting, so async wouldn't help
    there. But the Gemini API call genuinely benefits, since
    it's a real network request that takes 1-3 seconds.
    """
    if vectorstore is None:
        raise HTTPException(
            status_code=400,
            detail="No document uploaded yet. Call /upload first."
        )

    # Vector search - stays synchronous, it's fast/local
    relevant_chunks = vectorstore.similarity_search(
        request.question, k=request.k
    )

    context = "\n\n".join([
        f"[Source: {c.metadata.get('source', 'unknown')}]\n{c.page_content}"
        for c in relevant_chunks
    ])

    prompt = f"""You are a helpful AI assistant.
Answer using ONLY the provided context.
If the answer is not in the context, say "I don't have that information in the document."
Always mention which document your answer came from.

CONTEXT:
{context}

QUESTION: {request.question}

ANSWER:"""

    # THE KEY CHANGE: await + .aio.models instead of .models
    # This is the ASYNC version of the Gemini call —
    # it lets the server handle OTHER requests while waiting
    response = await gemini_client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=500,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
    )

    sources = [
        SourceInfo(
            chunk_id=c.metadata.get("chunk_id", i),
            source_file=c.metadata.get("source", "unknown"),
            preview=c.page_content[:100]
        )
        for i, c in enumerate(relevant_chunks)
    ]

    return AnswerResponse(
        question=request.question,
        answer=response.text.strip(),
        sources=sources,
        chunks_used=len(relevant_chunks)
    )


# =============================================
# ENDPOINT 3: Health Check
# =============================================

@app.get(
    "/health",
    tags=["System"],
    summary="Check API health status",
    description="Returns whether the API is running and models are loaded. Does not require authentication."
)
def health_check():
    """
    Simple endpoint to verify the API is running and
    models are loaded. Common pattern in production APIs
    for monitoring/uptime checks.
    """
    return {
        "status": "healthy",
        "models_loaded": embeddings_model is not None,
        "document_loaded": vectorstore is not None
    }