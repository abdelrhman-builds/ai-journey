# =============================================
# Day 16 - Embeddings + Chroma Vector Store
# Author: Abdelrhman
# Date: June 2026
# =============================================
# RAG Pipeline Steps covered today:
# Step 3: EMBED chunks (text → numbers)
# Step 4: STORE in Chroma vector database
# =============================================

import os
import math
import time
from dotenv import load_dotenv

# Document loading and splitting (from Day 15)
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chroma vector store
from langchain_chroma import Chroma

# HuggingFace local embeddings — FREE, no API key needed
# Downloads model (~90MB) on first run → runs locally forever after
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


# =============================================
# SECTION 1: Load and Split Document
# =============================================
# Repeat Steps 1-2 from Day 15
# In production this would be a separate ingestion pipeline

print("=" * 60)
print("SECTION 1: Load and Split Document")
print("=" * 60)

# TextLoader reads text file → returns list of Document objects
loader = TextLoader("sample_document.txt", encoding="utf-8")
documents = loader.load()
print(f"✅ Document loaded: {len(documents[0].page_content)} characters")

# Split into chunks of 500 chars with 50 char overlap
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # max characters per chunk
    chunk_overlap=50       # overlap preserves context at boundaries
)
chunks = splitter.split_documents(documents)
print(f"✅ Chunks created: {len(chunks)}")

# Add custom metadata to each chunk
# This metadata travels with chunks into the vector database
# Used later for citations: "Answer from chunk 3 of sample_document.txt"
for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = i
    chunk.metadata["source_file"] = "sample_document.txt"
    chunk.metadata["chunk_size"] = len(chunk.page_content)

print(f"✅ Metadata added to all {len(chunks)} chunks")


# =============================================
# SECTION 2: Initialize Embeddings Model
# =============================================
# HuggingFace sentence-transformers:
# → FREE — no API key or payment needed
# → LOCAL — runs on your machine, no internet after download
# → FAST — no rate limits, no waiting between calls
# → PRIVATE — your data never leaves your computer
# → all-MiniLM-L6-v2: lightweight model, 384-dimensional embeddings

print("\n" + "=" * 60)
print("SECTION 2: Embeddings Model")
print("=" * 60)

print("Loading local embeddings model...")
print("(First run downloads ~90MB — subsequent runs are instant)")

embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    # Other options:
    # "sentence-transformers/all-mpnet-base-v2" → higher quality, larger
    # "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" → Arabic support
    # "aubmindlab/bert-base-arabertv02" → best for Arabic text (Week 6)
)

print("✅ Local embeddings model loaded!")
print("   Model: all-MiniLM-L6-v2")
print("   Output: 384 numbers per text chunk")
print("   Cost: FREE — runs locally!")

# Test the embeddings model with a sample text
print("\nTesting embeddings on sample text...")
sample_text = "What is RAG in AI engineering?"
embedding_vector = embeddings_model.embed_query(sample_text)

print(f"Input:  '{sample_text}'")
print(f"Output: {len(embedding_vector)} numbers (384-dimensional vector)")
print(f"First 5 numbers: {[round(n, 4) for n in embedding_vector[:5]]}")
print(f"Last  5 numbers: {[round(n, 4) for n in embedding_vector[-5:]]}")


# =============================================
# SECTION 3: Semantic Similarity Demo
# =============================================
# Prove that similar texts have similar embedding numbers
# This is the CORE concept that makes RAG work

print("\n" + "=" * 60)
print("SECTION 3: Semantic Similarity Demo")
print("=" * 60)


def cosine_similarity(vec1, vec2):
    """
    Measures similarity between two embedding vectors.
    Returns value between -1 and 1:
    1.0  = identical meaning
    0.5  = somewhat similar
    0.0  = unrelated
    -1.0 = opposite meaning

    Formula: dot_product / (magnitude1 × magnitude2)
    """
    # Dot product: multiply each pair of numbers and sum
    dot_product = sum(a * b for a, b in zip(vec1, vec2))

    # Magnitude: sqrt of sum of squares
    magnitude1 = math.sqrt(sum(a ** 2 for a in vec1))
    magnitude2 = math.sqrt(sum(b ** 2 for b in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0

    return dot_product / (magnitude1 * magnitude2)


# Compare semantically similar vs different texts
text1 = "RAG stands for Retrieval Augmented Generation"
text2 = "Retrieval Augmented Generation helps AI answer accurately"
text3 = "I love eating pizza in Cairo on weekends"

print("Getting embeddings for 3 texts...")
emb1 = embeddings_model.embed_query(text1)
emb2 = embeddings_model.embed_query(text2)
emb3 = embeddings_model.embed_query(text3)

# Calculate similarities
sim_1_2 = cosine_similarity(emb1, emb2)
sim_1_3 = cosine_similarity(emb1, emb3)
sim_2_3 = cosine_similarity(emb2, emb3)

print(f"\nText 1: '{text1}'")
print(f"Text 2: '{text2}'")
print(f"Text 3: '{text3}'")

print(f"\nSimilarity Results:")
print(f"  Text 1 vs Text 2: {sim_1_2:.4f}  ← should be HIGH (both about RAG)")
print(f"  Text 1 vs Text 3: {sim_1_3:.4f}  ← should be LOW  (RAG vs pizza)")
print(f"  Text 2 vs Text 3: {sim_2_3:.4f}  ← should be LOW  (RAG vs pizza)")

print(f"\n💡 This is why RAG works:")
print(f"   User asks 'explain retrieval augmented generation'")
print(f"   → similar embedding to all RAG-related chunks")
print(f"   → those chunks retrieved automatically")
print(f"   → not pizza chunks retrieved!")


# =============================================
# SECTION 4: Create Chroma Vector Store
# =============================================
# Chroma stores our chunks AND their embeddings on disk
# persist_directory = local folder where database is saved
# Data survives Python restarts — build once, use forever

print("\n" + "=" * 60)
print("SECTION 4: Create Chroma Vector Store")
print("=" * 60)

# Remove existing database if it exists (fresh start)
persist_dir = "./chroma_db"

print(f"Creating vector store...")
print(f"Database location: {persist_dir}")

# Chroma.from_documents() does 3 things:
# 1. Takes text from each chunk
# 2. Calls embeddings_model to convert to numbers
# 3. Saves both text + numbers to disk at persist_dir
vectorstore = Chroma.from_documents(
    documents=chunks,               # our 8 text chunks
    embedding=embeddings_model,     # model to create embeddings
    persist_directory=persist_dir   # save to disk
)

print(f"\n✅ Vector store created successfully!")
print(f"✅ {len(chunks)} chunks embedded and stored")
print(f"✅ Database saved to: {persist_dir}")
print(f"✅ Data persists between Python restarts")


# =============================================
# SECTION 5: Similarity Search
# =============================================
# This is the MAGIC of RAG:
# Ask a question → get most relevant chunks back
# No keywords needed — pure semantic meaning search

print("\n" + "=" * 60)
print("SECTION 5: Similarity Search")
print("=" * 60)

# Test questions to search our document
questions = [
    "What is RAG?",
    "What are the career opportunities for AI engineers?",
    "What is a vector database?",
    "What programming skills do I need?"
]

for question in questions:
    print(f"\n🔍 Question: '{question}'")
    print("-" * 40)

    # similarity_search():
    # 1. Embeds the question
    # 2. Compares to all stored chunk embeddings
    # 3. Returns top k most similar chunks
    results = vectorstore.similarity_search(question, k=2)

    for i, result in enumerate(results):
        print(f"\n  [Result {i+1}]")
        # Show first 200 chars of each result
        print(f"  Content: {result.page_content[:200]}...")
        print(f"  Source: {result.metadata.get('source_file', 'unknown')}")
        print(f"  Chunk ID: {result.metadata.get('chunk_id', 'N/A')}")


# =============================================
# SECTION 6: Search With Similarity Scores
# =============================================

print("\n" + "=" * 60)
print("SECTION 6: Search With Scores")
print("=" * 60)

question = "How does RAG work step by step?"
print(f"Question: '{question}'")

# similarity_search_with_score() returns (document, score) tuples
# In Chroma: LOWER score = MORE similar (L2 distance)
results_with_scores = vectorstore.similarity_search_with_score(
    question, k=3
)

print(f"\nTop 3 results ranked by similarity:")
for i, (doc, score) in enumerate(results_with_scores):
    print(f"\n  [Rank {i+1}] Distance Score: {score:.4f}")
    print(f"  Content: {doc.page_content[:150]}...")

print("\n💡 Note: In Chroma, LOWER score = MORE similar")
print("   (Uses L2 distance, not cosine similarity)")


# =============================================
# SECTION 7: Load Existing Vector Store
# =============================================
# In production: build vector store ONCE (ingestion)
# Then LOAD it for each query (retrieval)
# Never rebuild unless documents change

print("\n" + "=" * 60)
print("SECTION 7: Load Existing Vector Store from Disk")
print("=" * 60)

# Load the already-built database from disk
# No need to re-embed — everything already stored
loaded_vs = Chroma(
    persist_directory=persist_dir,
    embedding_function=embeddings_model
)

# Test it works correctly
test_result = loaded_vs.similarity_search(
    "What skills does a junior AI engineer need?",
    k=1
)

print(f"✅ Vector store loaded from disk!")
print(f"✅ Test search returned: {len(test_result)} result(s)")
print(f"Preview: {test_result[0].page_content[:100]}...")


# =============================================
# DAY 16 SUMMARY
# =============================================

print(f"\n{'='*60}")
print(f"DAY 16 COMPLETE — SUMMARY")
print(f"{'='*60}")
print(f"✅ Embeddings: sentence-transformers/all-MiniLM-L6-v2")
print(f"✅ Each chunk → 384 numbers (embedding vector)")
print(f"✅ Semantic similarity proven:")
print(f"   Similar texts → similar numbers")
print(f"   Different texts → different numbers")
print(f"✅ {len(chunks)} chunks stored in Chroma vector database")
print(f"✅ Similarity search working correctly")
print(f"✅ Database persisted to disk → survives restarts")
print(f"\n📋 RAG Pipeline Progress:")
print(f"   ✅ Step 1: LOAD document")
print(f"   ✅ Step 2: SPLIT into chunks")
print(f"   ✅ Step 3: EMBED chunks")
print(f"   ✅ Step 4: STORE in Chroma")
print(f"   ⏳ Steps 5-9: Full pipeline → Day 17!")
print(f"\nReady for Day 17: Complete RAG Q&A System!")