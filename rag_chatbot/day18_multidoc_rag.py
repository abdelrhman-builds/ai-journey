# =============================================
# Day 18 - Multi-Document RAG with Source Attribution
# Author: Abdelrhman
# Date: June 2026
# =============================================
# New concepts beyond Day 17:
# 1. Loading multiple documents into ONE Chroma collection
# 2. Source metadata tagging (which doc each chunk came from)
# 3. Metadata filtering (search within specific documents)
# 4. Source attribution in final answers
#
# NOTE: Developed and tested in Google Colab due to
# Windows compatibility issue with embedding libraries.
# =============================================

import os
import time
import google.genai as genai
from google.genai import types
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


# =============================================
# SECTION 1: Sample Multi-Document Dataset
# =============================================
# In production: these would be real PDFs, Word docs, etc.
# loaded via PyPDFLoader, Docx2txtLoader, DirectoryLoader

DOC1_TEXT = """AI Engineering Course — Complete Guide

CHAPTER 1: WHAT IS AI ENGINEERING?
AI Engineering is the practice of building production-ready
artificial intelligence applications. AI engineers focus on
deploying AI systems that work reliably in the real world.

CHAPTER 2: WHAT IS RAG?
RAG stands for Retrieval Augmented Generation. It combines
document retrieval with language model generation to produce
accurate, grounded answers.

CHAPTER 6: CAREER PATH
Junior AI Engineer skills required: Python, LLM API usage,
RAG implementation, FastAPI, Git. Salary range in Egypt:
15,000-25,000 EGP for juniors, 60,000-100,000 EGP for seniors."""

DOC2_TEXT = """TechCorp Egypt — Employee Handbook

SECTION 1: WORKING HOURS
Standard working hours are 9 AM to 5 PM, Sunday to Thursday.
Remote work is allowed up to 2 days per week with manager approval.

SECTION 2: LEAVE POLICY
Annual leave: 21 days per year for employees with less than 5 years.
Sick leave: 14 days per year, requires medical certificate after 2 days.
Maternity leave: 90 days fully paid.

SECTION 3: SALARY AND BENEFITS
Salaries are paid monthly via bank transfer on the 27th of each month.
Health insurance covers employee and immediate family members.
Annual bonus based on performance review, typically 1-2 months salary.

SECTION 4: REMOTE WORK POLICY
Employees may request remote work arrangements.
Must maintain availability during core hours 10 AM - 3 PM.
Equipment provided: laptop, monitor, ergonomic chair allowance."""

DOC3_TEXT = """CloudSync Pro — User Manual

GETTING STARTED
CloudSync Pro is a file synchronization tool for teams.
Install the desktop app from cloudsync.com/download.
Create an account using your work email address.

FEATURES
Real-time sync across all devices.
Version history for the last 30 days.
Team sharing with permission levels: Viewer, Editor, Admin.
Automatic backup every 6 hours.

TROUBLESHOOTING
If sync fails, check your internet connection first.
Clear cache: Settings > Advanced > Clear Cache.
Contact support at support@cloudsync.com for persistent issues.

PRICING
Free tier: 5GB storage, 1 user.
Pro tier: 100GB storage, 5 users, $9.99/month.
Enterprise: Custom pricing, unlimited storage."""


# =============================================
# SECTION 2: Build Multi-Document Vector Store
# =============================================

def build_multi_doc_vectorstore():
    """
    Loads multiple documents, splits them, and stores
    all chunks in ONE Chroma collection with source metadata.

    In production, replace the text blocks above with:
        from langchain_community.document_loaders import DirectoryLoader
        loader = DirectoryLoader("./documents/", glob="**/*.*")
        documents = loader.load()
    """

    # Step 1: Create Document objects with source metadata
    # Each document gets tagged with WHERE it came from
    # This metadata is what enables filtering and citations later
    documents = [
        Document(
            page_content=DOC1_TEXT,
            metadata={"source": "ai_course.txt", "category": "education"}
        ),
        Document(
            page_content=DOC2_TEXT,
            metadata={"source": "hr_policy.txt", "category": "hr"}
        ),
        Document(
            page_content=DOC3_TEXT,
            metadata={"source": "product_manual.txt", "category": "product"}
        )
    ]

    print(f"✅ {len(documents)} documents created with metadata")
    for doc in documents:
        print(f"  - {doc.metadata['source']} ({doc.metadata['category']})")

    # Step 2: Split ALL documents into chunks at once
    # split_documents() handles multiple Document objects in one call
    # Each resulting chunk AUTOMATICALLY inherits its parent's metadata
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )
    all_chunks = splitter.split_documents(documents)

    # Add chunk-level metadata on top of inherited document metadata
    for i, chunk in enumerate(all_chunks):
        chunk.metadata["chunk_id"] = i

    print(f"\n✅ Total chunks created: {len(all_chunks)}")
    for source in ["ai_course.txt", "hr_policy.txt", "product_manual.txt"]:
        count = sum(1 for c in all_chunks if c.metadata["source"] == source)
        print(f"  {source}: {count} chunks")

    # Step 3: Initialize embeddings model
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("\n✅ Embeddings model loaded!")

    # Step 4: Store ALL chunks from ALL documents in ONE collection
    # This is the key difference from Day 17 (single document)
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings_model,
        persist_directory="./multi_doc_chroma"
    )
    print(f"✅ Vector store created with {len(all_chunks)} chunks from {len(documents)} documents!")

    return vectorstore


# =============================================
# SECTION 3: Multi-Document RAG with Citations
# =============================================

def ask_multi_doc_rag(question, vectorstore, client, k=3, source_filter=None):
    """
    RAG with multi-document support and source citation.

    Args:
        question:      User's question
        vectorstore:   Chroma store containing multiple documents
        client:        Gemini client
        k:             Number of chunks to retrieve
        source_filter: Optional dict to restrict search to one document
                       e.g. {"source": "hr_policy.txt"}
                       or by category: {"category": "hr"}

    Returns:
        Dictionary with answer and which document(s) it came from
    """
    # Search with optional metadata filter
    # Filtering restricts the search space BEFORE similarity ranking
    if source_filter:
        chunks = vectorstore.similarity_search(question, k=k, filter=source_filter)
    else:
        chunks = vectorstore.similarity_search(question, k=k)

    # Build context with explicit document labels
    # This helps the LLM know which chunk came from which source
    context = "\n\n".join([
        f"[Document: {c.metadata['source']}]\n{c.page_content}"
        for c in chunks
    ])

    # Track which unique documents contributed to this answer
    sources_used = list(set(c.metadata['source'] for c in chunks))

    # RAG prompt with multi-document awareness
    prompt = f"""You are a helpful AI assistant with access to multiple documents.
Answer using ONLY the provided context.
If the answer is not in the context, say "I don't have that information."
Always state which document(s) your answer came from.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=500,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
    )

    return {
        "question": question,
        "answer": response.text.strip(),
        "documents_used": sources_used,
        "chunks_used": len(chunks)
    }


# =============================================
# SECTION 4: Demonstration — Filtering Examples
# =============================================

def demo_filtering(vectorstore):
    """Shows the difference between filtered and unfiltered search"""

    print("\n" + "=" * 60)
    print("FILTERING DEMONSTRATION")
    print("=" * 60)

    question = "How much does the pro plan cost?"

    # Without filter — searches ALL documents
    print(f"\n🔍 WITHOUT filter: '{question}'")
    unfiltered = vectorstore.similarity_search(question, k=2)
    for r in unfiltered:
        print(f"  Source: {r.metadata['source']}")

    # With filter — restricts to ONE document only
    print(f"\n🔍 WITH filter (source=hr_policy.txt): '{question}'")
    filtered = vectorstore.similarity_search(
        question, k=2, filter={"source": "hr_policy.txt"}
    )
    for r in filtered:
        print(f"  Source: {r.metadata['source']} (forced)")

    print("\n💡 Filtering forces search into a specific document")
    print("   Useful when user selects 'search only in HR docs'")


# =============================================
# MAIN
# =============================================

if __name__ == "__main__":
    # Initialize Gemini
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    print("✅ Gemini client initialized!")

    # Build multi-document vector store
    print("\nBuilding multi-document vector store...")
    vectorstore = build_multi_doc_vectorstore()

    # Run filtering demonstration
    demo_filtering(vectorstore)

    # Test cross-document questions
    print("\n" + "=" * 60)
    print("MULTI-DOCUMENT RAG TEST")
    print("=" * 60)

    test_questions = [
        "What is the leave policy and how many days do I get?",
        "How much does CloudSync Pro cost per month?",
        "What is RAG and what skills do I need to learn it?",
        "What is the meaning of life?"  # should fail gracefully
    ]

    for q in test_questions:
        print(f"\n{'='*50}")
        print(f"Q: {q}")
        result = ask_multi_doc_rag(q, vectorstore, client)
        print(f"A: {result['answer']}")
        print(f"📄 Documents used: {result['documents_used']}")
        time.sleep(13)

    print("\n✅ Day 18 Complete — Multi-document RAG working!")