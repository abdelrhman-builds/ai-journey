# =============================================
# Day 15 - RAG Foundations
# Author: Abdelrhman
# Date: June 2026
# =============================================
# Today we cover Steps 1-2 of the RAG pipeline:
# Step 1: LOAD document
# Step 2: SPLIT into chunks
# =============================================

import os
from dotenv import load_dotenv

# LangChain document loaders
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader

# LangChain text splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


# =============================================
# SECTION 1: Load a Text Document
# =============================================

print("=" * 60)
print("SECTION 1: Document Loading")
print("=" * 60)

# Load our sample text document
# TextLoader reads the file and returns a list of Document objects
loader = TextLoader("sample_document.txt", encoding="utf-8")
documents = loader.load()

print(f"\nDocument loaded successfully!")
print(f"Number of documents: {len(documents)}")
print(f"Content length: {len(documents[0].page_content)} characters")
print(f"\nFirst 200 characters:")
print(documents[0].page_content[:200])

# A Document object has two parts:
# 1. page_content → the actual text
# 2. metadata     → information about the document
print(f"\nDocument metadata: {documents[0].metadata}")


# =============================================
# SECTION 2: Split Into Chunks
# =============================================

print("\n" + "=" * 60)
print("SECTION 2: Text Splitting")
print("=" * 60)

# RecursiveCharacterTextSplitter is the best splitter for most use cases
# It tries to split on paragraphs first, then sentences, then words
# This keeps related content together

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # each chunk = max 500 characters
    chunk_overlap=50,      # 50 characters overlap between chunks
                           # overlap prevents losing context at boundaries
    length_function=len,   # how to measure chunk size
)

# Split our document into chunks
chunks = splitter.split_documents(documents)

print(f"\nTotal chunks created: {len(chunks)}")
print(f"Chunk size setting: 500 characters")
print(f"Chunk overlap: 50 characters")

# Show first 3 chunks
print("\n--- First 3 Chunks ---")
for i, chunk in enumerate(chunks[:3]):
    print(f"\n[Chunk {i+1}]")
    print(f"Length: {len(chunk.page_content)} characters")
    print(f"Content: {chunk.page_content}")
    print(f"Metadata: {chunk.metadata}")
    print("-" * 40)


# =============================================
# SECTION 3: Experiment With Chunk Sizes
# =============================================

print("\n" + "=" * 60)
print("SECTION 3: Chunk Size Experiments")
print("=" * 60)

# Experiment with different chunk sizes
chunk_sizes = [200, 500, 1000]

for size in chunk_sizes:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=20
    )
    chunks_test = splitter.split_documents(documents)
    print(f"Chunk size {size}: {len(chunks_test)} chunks created")

print("\n💡 Key insight:")
print("Smaller chunks → more chunks → more precise retrieval")
print("Larger chunks  → fewer chunks → more context per chunk")
print("Sweet spot: 500-1000 characters for most documents")


# =============================================
# SECTION 4: Load a Real PDF (Optional Test)
# =============================================

print("\n" + "=" * 60)
print("SECTION 4: PDF Loading")
print("=" * 60)

# Check if there's a PDF in the documents folder
pdf_path = "documents\Lecture 1 NLP Preprocessing.pdf"

if os.path.exists(pdf_path):
    pdf_loader = PyPDFLoader(pdf_path)
    pdf_docs = pdf_loader.load()
    print(f"PDF loaded: {len(pdf_docs)} pages")
    print(f"First page preview: {pdf_docs[0].page_content[:200]}")
else:
    print("No PDF found in documents/ folder")
    print("→ Place any PDF there to test PDF loading")
    print("→ Tomorrow we'll add PDF support to the full pipeline")


# =============================================
# SECTION 5: Understanding Document Objects
# =============================================

print("\n" + "=" * 60)
print("SECTION 5: Document Object Structure")
print("=" * 60)

# Use our 500-char splitter for final chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
final_chunks = splitter.split_documents(documents)

print(f"\nTotal chunks: {len(final_chunks)}")
print(f"\nDocument object structure:")
print(f"  chunk.page_content → the text")
print(f"  chunk.metadata     → source info")

# Add custom metadata to chunks
print(f"\nAdding custom metadata to chunks...")
for i, chunk in enumerate(final_chunks):
    chunk.metadata["chunk_id"] = i
    chunk.metadata["chunk_size"] = len(chunk.page_content)
    chunk.metadata["source_file"] = "sample_document.txt"

print(f"Sample chunk metadata after update:")
print(f"  {final_chunks[0].metadata}")

# Summary
print(f"\n{'='*60}")
print(f"DAY 15 SUMMARY")
print(f"{'='*60}")
print(f"✅ Document loaded: sample_document.txt")
print(f"✅ Chunks created: {len(final_chunks)}")
print(f"✅ Average chunk size: {sum(len(c.page_content) for c in final_chunks) // len(final_chunks)} chars")
print(f"✅ Metadata added to all chunks")
print(f"\nReady for Day 16: Embeddings + Vector Store!")