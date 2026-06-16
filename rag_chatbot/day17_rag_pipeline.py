# =============================================
# Day 17 - Full RAG Pipeline
# Author: Abdelrhman
# Date: June 2026
# =============================================
# Complete RAG Pipeline — Steps 5-9:
# Step 5: User asks question
# Step 6: Embed question (auto by vectorstore)
# Step 7: Search Chroma → find relevant chunks
# Step 8: Build prompt with context + question
# Step 9: Send to Gemini → get grounded answer
# =============================================

import os
import time
import google.genai as genai
from google.genai import types
from dotenv import load_dotenv

# Document processing (Days 15-16)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


# =============================================
# SETUP: Document + Vector Store
# =============================================

def setup_vectorstore():
    """
    Complete ingestion pipeline:
    Load → Split → Embed → Store
    Returns ready-to-use vector store
    """
    # Sample document (in production: load from PDF)
    sample_text = """AI Engineering Course — Complete Guide

    CHAPTER 1: WHAT IS AI ENGINEERING?
    AI Engineering is the practice of building production-ready
    artificial intelligence applications. Unlike data scientists
    who focus on research and model development, AI engineers
    focus on deploying AI systems that work reliably in the real world.

    Key responsibilities of an AI Engineer:
    - Building LLM-powered applications
    - Implementing RAG systems for document search
    - Deploying AI models as APIs using FastAPI
    - Fine-tuning models for specific domains
    - Building AI agents with tool use

    CHAPTER 2: WHAT IS RAG?
    RAG stands for Retrieval Augmented Generation.
    It is a technique that combines document retrieval with
    language model generation to produce accurate, grounded answers.

    How RAG works:
    1. Documents are loaded and split into chunks
    2. Chunks are converted to embeddings (numerical vectors)
    3. Embeddings are stored in a vector database
    4. User asks a question
    5. Question is converted to an embedding
    6. Similar chunks are retrieved from the vector database
    7. Chunks + question are sent to the LLM
    8. LLM generates an answer based on retrieved chunks

    Benefits of RAG:
    - Reduces hallucination (AI only uses real document content)
    - Knowledge can be updated without retraining
    - Sources can be cited for every answer
    - Works with any domain-specific documents

    CHAPTER 3: VECTOR DATABASES
    A vector database stores embeddings and enables similarity search.
    Popular vector databases include Chroma, Pinecone, and Weaviate.

    Chroma is the best choice for local development because:
    - Runs completely locally (no API key needed)
    - Easy to set up with pip install chromadb
    - Persistent storage (data survives restarts)
    - Fast similarity search

    CHAPTER 4: EMBEDDINGS
    Embeddings are numerical representations of text.
    Similar texts have similar embeddings.
    This enables semantic search — finding meaning, not just keywords.

    CHAPTER 5: THE AI ENGINEERING STACK
    A complete AI engineering stack includes:
    - Python for core programming
    - LangChain for AI application orchestration
    - Gemini API for language model inference
    - Chroma for vector storage
    - FastAPI for serving as an API
    - Streamlit for building demos quickly
    - Docker for containerization
    - Git and GitHub for version control

    CHAPTER 6: CAREER PATH
    Junior AI Engineer skills required:
    - Python proficiency
    - LLM API usage (OpenAI, Gemini, Anthropic)
    - RAG system implementation
    - Basic FastAPI knowledge
    - Git and GitHub
    - 2-3 deployed projects

    Salary ranges in Egypt (2026):
    - Junior AI Engineer: 15,000 - 25,000 EGP
    - Mid-level AI Engineer: 30,000 - 50,000 EGP
    - Senior AI Engineer: 60,000 - 100,000 EGP
    - Remote (international): $2,000 - $8,000 USD"""

    # Create Document object
    documents = [Document(
        page_content=sample_text,
        metadata={"source": "sample_document.txt"}
    )]

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["source_file"] = "sample_document.txt"

    # Initialize local embeddings
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create Chroma vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory="./chroma_db"
    )

    return vectorstore, len(chunks)


# =============================================
# CORE RAG FUNCTION
# =============================================

def ask_rag(question, vectorstore, client, k=3):
    """
    Complete RAG pipeline in one function.

    Args:
        question:    User's question
        vectorstore: Chroma vector store
        client:      Gemini client
        k:           Number of chunks to retrieve

    Returns:
        Dictionary with answer and sources
    """
    # Step 1: Search vector store
    # Automatically embeds question + finds similar chunks
    relevant_chunks = vectorstore.similarity_search(question, k=k)

    # Step 2: Build context from chunks
    context = "\n\n".join([
        f"[Chunk {chunk.metadata.get('chunk_id', i)}]\n{chunk.page_content}"
        for i, chunk in enumerate(relevant_chunks)
    ])

    # Step 3: RAG prompt — most important pattern
    rag_prompt = f"""You are a helpful AI assistant.
Answer the question using ONLY the provided context.
If the answer is not in the context, say exactly:
"I don't have that information in the document."
Always mention which chunk your answer came from.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    # Step 4: Send to Gemini
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=rag_prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=500,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
    )

    return {
        "question": question,
        "answer": response.text.strip(),
        "chunks_used": len(relevant_chunks),
        "sources": [
            {
                "chunk_id": chunk.metadata.get('chunk_id', i),
                "preview": chunk.page_content[:100],
                "source_file": chunk.metadata.get('source_file', 'unknown')
            }
            for i, chunk in enumerate(relevant_chunks)
        ]
    }


# =============================================
# INTERACTIVE CHAT
# =============================================

def run_rag_chat(vectorstore, client):
    """Interactive RAG Q&A system"""
    print("=" * 60)
    print("   📚 RAG Document Q&A System")
    print("=" * 60)
    print("Ask questions about the AI Engineering document.")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() == 'quit':
            print("Goodbye! 🚀")
            break

        try:
            result = ask_rag(question, vectorstore, client)
            print(f"\n📖 Answer: {result['answer']}")
            print(f"📚 Sources: Chunks {[s['chunk_id'] for s in result['sources']]}\n")
            time.sleep(13)
        except Exception as e:
            print(f"Error: {e}")


# =============================================
# MAIN
# =============================================

if __name__ == "__main__":
    # Initialize Gemini
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    print("✅ Gemini client initialized!")

    # Setup vector store
    print("\nSetting up vector store...")
    vectorstore, chunk_count = setup_vectorstore()
    print(f"✅ Vector store ready with {chunk_count} chunks!")

    # Run interactive chat
    run_rag_chat(vectorstore, client)