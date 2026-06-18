# =============================================
# Day 19 - Streamlit RAG Web App
# Author: Abdelrhman
# Date: June 2026
# =============================================

import os
import time
import streamlit as st
import google.genai as genai
from google.genai import types
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()


# =============================================
# PAGE CONFIG — must be first Streamlit command
# =============================================

st.set_page_config(
    page_title="AI Document Q&A",
    page_icon="📚",
    layout="wide"
)


# =============================================
# CACHED RESOURCES
# =============================================
# @st.cache_resource prevents reloading the embedding model
# every time the user interacts with the app (saves time)

@st.cache_resource
def load_embeddings_model():
    """Load embeddings model once, reuse across app reruns"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def get_gemini_client():
    """Initialize Gemini client once"""
    api_key = os.getenv("GEMINI_API_KEY")
    return genai.Client(api_key=api_key)


# =============================================
# SESSION STATE — persists across reruns
# =============================================
# Streamlit reruns the ENTIRE script on every interaction
# session_state lets us keep data between those reruns

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = []


# =============================================
# CORE FUNCTIONS
# =============================================

def process_uploaded_files(uploaded_files, embeddings_model):
    """
    Processes uploaded files into a searchable vector store.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects
        embeddings_model: The embeddings model to use

    Returns:
        Chroma vectorstore ready for searching
    """
    all_documents = []

    for uploaded_file in uploaded_files:
        # Save uploaded file temporarily to read it
        temp_path = f"./temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load based on file type
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
        else:
            # Treat as text file
            with open(temp_path, "r", encoding="utf-8") as f:
                text = f.read()
            docs = [Document(page_content=text, metadata={})]

        # Tag every document with its source filename
        for doc in docs:
            doc.metadata["source"] = uploaded_file.name

        all_documents.extend(docs)

        # Clean up temp file
        os.remove(temp_path)

    # Split all documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(all_documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory="./streamlit_chroma_db"
    )

    return vectorstore, len(chunks)


def ask_rag(question, vectorstore, client, k=3):
    """Core RAG function — search + prompt + answer"""

    relevant_chunks = vectorstore.similarity_search(question, k=k)

    context = "\n\n".join([
        f"[Source: {c.metadata.get('source', 'unknown')}]\n{c.page_content}"
        for c in relevant_chunks
    ])

    sources_used = list(set(c.metadata.get('source', 'unknown') for c in relevant_chunks))

    prompt = f"""You are a helpful AI assistant.
Answer using ONLY the provided context.
If the answer is not in the context, say "I don't have that information in the document."
Always mention which document your answer came from.

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
        "answer": response.text.strip(),
        "sources": sources_used
    }


# =============================================
# UI — SIDEBAR (File Upload)
# =============================================

with st.sidebar:
    st.header("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF or text files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("Process Documents", type="primary"):
        with st.spinner("Processing documents... this may take a minute"):
            embeddings_model = load_embeddings_model()
            vectorstore, chunk_count = process_uploaded_files(
                uploaded_files, embeddings_model
            )
            st.session_state.vectorstore = vectorstore
            st.session_state.documents_loaded = [f.name for f in uploaded_files]

        st.success(f"✅ Processed {len(uploaded_files)} documents into {chunk_count} chunks!")

    # Show currently loaded documents
    if st.session_state.documents_loaded:
        st.subheader("📄 Loaded Documents")
        for doc_name in st.session_state.documents_loaded:
            st.write(f"• {doc_name}")

    st.divider()
    if st.button("🗑️ Clear All"):
        st.session_state.vectorstore = None
        st.session_state.chat_history = []
        st.session_state.documents_loaded = []
        st.rerun()


# =============================================
# UI — MAIN AREA (Chat Interface)
# =============================================

st.title("📚 AI Document Q&A System")
st.caption("Upload documents and ask questions — powered by RAG")

if st.session_state.vectorstore is None:
    st.info("👈 Upload documents in the sidebar to get started")
else:
    # Display chat history
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.write(entry["answer"])
            st.caption(f"📄 Sources: {', '.join(entry['sources'])}")

    # Question input
    question = st.chat_input("Ask a question about your documents...")

    if question:
        # Display user question immediately
        with st.chat_message("user"):
            st.write(question)

        # Get and display answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                client = get_gemini_client()
                result = ask_rag(question, st.session_state.vectorstore, client)
                st.write(result["answer"])
                st.caption(f"📄 Sources: {', '.join(result['sources'])}")

        # Save to history
        st.session_state.chat_history.append({
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"]
        })