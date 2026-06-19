# =============================================
# AI Document Q&A — RAG System (Secured Version)
# Deployed on HuggingFace Spaces
# Author: Abdelrhman
# =============================================
# Security layers implemented:
# 1. File type whitelist (PDF, TXT only)
# 2. File size limit (5MB max per file)
# 3. File signature validation (real PDF check)
# 4. Max files per upload (3 max)
# 5. Daily API quota protection
# 6. Per-session question limit
# 7. Full error handling (no crashes from bad files)
# 8. Automatic temp file cleanup
# =============================================

import os
import datetime
import streamlit as st
import google.genai as genai
from google.genai import types

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader


# =============================================
# CONFIGURATION
# =============================================

MAX_FILE_SIZE_MB = 5
MAX_FILES_PER_UPLOAD = 3
MAX_QUESTIONS_PER_SESSION = 5
DAILY_API_LIMIT = 18          # buffer below 20/day free tier quota
COUNTER_FILE = "daily_counter.txt"


# =============================================
# PAGE CONFIG
# =============================================

st.set_page_config(
    page_title="AI Document Q&A",
    page_icon="📚",
    layout="wide"
)


# =============================================
# CACHED RESOURCES
# =============================================

@st.cache_resource
def load_embeddings_model():
    """Load embeddings model once, reuse across all reruns."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def get_gemini_client():
    """Initialize Gemini client once. Key comes from HF Space Secrets."""
    api_key = os.environ.get("GEMINI_API_KEY")
    return genai.Client(api_key=api_key)


# =============================================
# SESSION STATE
# =============================================

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = []

if "question_count" not in st.session_state:
    st.session_state.question_count = 0


# =============================================
# SECURITY: DAILY QUOTA TRACKING
# =============================================
# Protects total API usage across ALL visitors, not just one session.
# Uses a simple file on disk since Streamlit session_state is
# per-browser-session and can't track usage across different users.

def check_daily_limit():
    """
    Reads today's question count from disk.
    Resets automatically when the date changes.
    """
    today = datetime.date.today().isoformat()

    if os.path.exists(COUNTER_FILE):
        try:
            with open(COUNTER_FILE, "r") as f:
                data = f.read().strip().split(",")
                stored_date, count = data[0], int(data[1])
        except (ValueError, IndexError):
            stored_date, count = today, 0
    else:
        stored_date, count = today, 0

    # Reset counter if it's a new day
    if stored_date != today:
        stored_date, count = today, 0

    return stored_date, count


def increment_daily_count(stored_date, count):
    """Saves the updated count back to disk."""
    with open(COUNTER_FILE, "w") as f:
        f.write(f"{stored_date},{count + 1}")


# =============================================
# SECURITY: FILE VALIDATION
# =============================================

def validate_file(uploaded_file):
    """
    Validates an uploaded file against size and content rules.

    Returns:
        (is_valid: bool, error_message: str)
    """
    # Check file size
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        return False, (f"{uploaded_file.name} is too large "
                        f"({file_size_mb:.1f}MB). Max size: {MAX_FILE_SIZE_MB}MB")

    # Check actual file signature for PDFs
    # Prevents someone renaming a non-PDF file to .pdf
    if uploaded_file.name.lower().endswith(".pdf"):
        header = uploaded_file.getvalue()[:4]
        if header != b'%PDF':
            return False, (f"{uploaded_file.name} is not a valid PDF "
                            f"(failed signature check)")

    # Check file isn't empty
    if uploaded_file.size == 0:
        return False, f"{uploaded_file.name} is empty"

    return True, "OK"


# =============================================
# CORE FUNCTIONS
# =============================================

def process_uploaded_files(uploaded_files, embeddings_model):
    """
    Validates and processes uploaded files into a searchable vector store.
    Skips invalid files instead of crashing the whole batch.
    """
    all_documents = []
    skipped_files = []

    for uploaded_file in uploaded_files:
        # SECURITY: validate before doing anything else
        is_valid, message = validate_file(uploaded_file)
        if not is_valid:
            skipped_files.append(message)
            continue

        temp_path = f"./temp_{uploaded_file.name}"
        try:
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if uploaded_file.name.lower().endswith(".pdf"):
                loader = PyPDFLoader(temp_path)
                docs = loader.load()
            else:
                with open(temp_path, "r", encoding="utf-8") as f:
                    text = f.read()
                docs = [Document(page_content=text, metadata={})]

            for doc in docs:
                doc.metadata["source"] = uploaded_file.name

            all_documents.extend(docs)

        except Exception as e:
            skipped_files.append(f"{uploaded_file.name}: failed to process ({str(e)[:100]})")
        finally:
            # SECURITY: always clean up temp files, even on error
            if os.path.exists(temp_path):
                os.remove(temp_path)

    if skipped_files:
        for msg in skipped_files:
            st.warning(f"⚠️ Skipped: {msg}")

    if not all_documents:
        return None, 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory="./streamlit_chroma_db"
    )

    return vectorstore, len(chunks)


def ask_rag(question, vectorstore, client, k=3):
    """Core RAG function — search + build prompt + get answer."""
    relevant_chunks = vectorstore.similarity_search(question, k=k)

    context = "\n\n".join([
        f"[Source: {c.metadata.get('source', 'unknown')}]\n{c.page_content}"
        for c in relevant_chunks
    ])

    sources_used = list(set(
        c.metadata.get('source', 'unknown') for c in relevant_chunks
    ))

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

    return {"answer": response.text.strip(), "sources": sources_used}


# =============================================
# UI — SIDEBAR (File Upload Area)
# =============================================

with st.sidebar:
    st.header("📁 Upload Documents")
    st.caption(f"Max {MAX_FILES_PER_UPLOAD} files, {MAX_FILE_SIZE_MB}MB each")

    uploaded_files = st.file_uploader(
        "Choose PDF or text files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    # SECURITY: enforce max files per upload
    if uploaded_files and len(uploaded_files) > MAX_FILES_PER_UPLOAD:
        st.error(f"❌ Maximum {MAX_FILES_PER_UPLOAD} files allowed. "
                  f"Only the first {MAX_FILES_PER_UPLOAD} will be processed.")
        uploaded_files = uploaded_files[:MAX_FILES_PER_UPLOAD]

    if uploaded_files and st.button("Process Documents", type="primary"):
        # SECURITY: wrap in try/except so a bad file can't crash the app
        try:
            with st.spinner("Processing documents... this may take a minute"):
                embeddings_model = load_embeddings_model()
                vectorstore, chunk_count = process_uploaded_files(
                    uploaded_files, embeddings_model
                )

            if vectorstore is None:
                st.error("❌ No valid documents could be processed. "
                          "Please check the file format and try again.")
            else:
                st.session_state.vectorstore = vectorstore
                st.session_state.documents_loaded = [f.name for f in uploaded_files]
                st.success(f"✅ Processed into {chunk_count} chunks!")

        except Exception as e:
            st.error(f"❌ Failed to process documents. "
                      f"Error: {str(e)[:150]}")

    if st.session_state.documents_loaded:
        st.subheader("📄 Loaded Documents")
        for doc_name in st.session_state.documents_loaded:
            st.write(f"• {doc_name}")

    st.divider()
    if st.button("🗑️ Clear All"):
        st.session_state.vectorstore = None
        st.session_state.chat_history = []
        st.session_state.documents_loaded = []
        st.session_state.question_count = 0
        st.rerun()


# =============================================
# UI — MAIN AREA (Chat Interface)
# =============================================

st.title("📚 AI Document Q&A System")
st.caption("Upload documents and ask questions — powered by RAG | Built by Abdelrhman")

if st.session_state.vectorstore is None:
    st.info("👈 Upload documents in the sidebar to get started")
else:
    # Replay chat history
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.write(entry["answer"])
            st.caption(f"📄 Sources: {', '.join(entry['sources'])}")

    # Show remaining questions for this session
    remaining = MAX_QUESTIONS_PER_SESSION - st.session_state.question_count
    st.caption(f"💬 Demo limit: {max(remaining, 0)} questions remaining this session")

    question = st.chat_input("Ask a question about your documents...")

    if question:
        # SECURITY CHECK 1: per-session limit (prevents one user spamming)
        if st.session_state.question_count >= MAX_QUESTIONS_PER_SESSION:
            st.warning(
                "⚠️ Demo limit reached for this session. "
                "Refresh the page to start a new session, or message "
                "me on LinkedIn for a full walkthrough of this project."
            )
        else:
            # SECURITY CHECK 2: global daily quota (protects total API usage)
            stored_date, current_count = check_daily_limit()

            if current_count >= DAILY_API_LIMIT:
                st.error(
                    "⚠️ This demo has reached its daily usage limit "
                    "(shared free-tier API quota). Please try again "
                    "tomorrow, or contact me on LinkedIn for a live demo."
                )
            else:
                with st.chat_message("user"):
                    st.write(question)

                with st.chat_message("assistant"):
                    try:
                        with st.spinner("Thinking..."):
                            client = get_gemini_client()
                            result = ask_rag(question, st.session_state.vectorstore, client)
                            st.write(result["answer"])
                            st.caption(f"📄 Sources: {', '.join(result['sources'])}")

                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": result["answer"],
                            "sources": result["sources"]
                        })
                        st.session_state.question_count += 1
                        increment_daily_count(stored_date, current_count)

                    except Exception as e:
                        st.error(f"❌ Something went wrong: {str(e)[:150]}")