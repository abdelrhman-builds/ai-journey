# =============================================
# Day 34 - Multi-format RAG agent
# Author: Abdelrhman
# Date: July 2026
# =============================================
# One agent, two information sources:
# 1. RAG tool: searches uploaded documents
# 2. Web search tool: searches the live internet
# Agent decides which to use based on the question.
# =============================================

import os
import time
import tempfile
from datetime import date
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.prebuilt import create_react_agent
from tavily import TavilyClient

import google.genai as genai
from google.genai import types

load_dotenv()


# =============================================
# SECTION 1: Initialize Shared Resources
# =============================================
# All expensive initializations done ONCE at module
# level — same pattern established on Day 23's FastAPI
# startup event and maintained throughout Week 5.

tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

print("Loading embeddings model...")
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("✅ Embeddings model loaded!")

# Global vector store — populated when a document is loaded
vectorstore = None


# =============================================
# SECTION 2: RAG Setup Tool (Upgraded - Multi-Format)
# =============================================

import tempfile

@tool
def load_document(file_content_b64: str, filename: str) -> str:
    """
    Load a document into the RAG system. Supports PDF, TXT, DOCX, XLSX, CSV.
    file_content_b64: base64 encoded file content
    filename: original filename including extension (e.g. report.pdf)
    """
    import base64
    import tempfile
    import os

    global vectorstore

    # Decode the file content
    file_bytes = base64.b64decode(file_content_b64)

    # Save to a temp file so loaders can read it
    suffix = os.path.splitext(filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        # Route to correct loader based on file extension
        if suffix == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()

        elif suffix == ".txt":
            with open(tmp_path, "r", encoding="utf-8") as f:
                text = f.read()
            documents = [Document(page_content=text,
                                  metadata={"source": filename})]

        elif suffix == ".docx":
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(tmp_path)
            documents = loader.load()

        elif suffix == ".csv":
            from langchain_community.document_loaders import CSVLoader
            loader = CSVLoader(tmp_path)
            documents = loader.load()

        elif suffix in [".xlsx", ".xls"]:
            from langchain_community.document_loaders import UnstructuredExcelLoader
            loader = UnstructuredExcelLoader(tmp_path)
            documents = loader.load()

        else:
            return f"Unsupported file type: {suffix}. Supported: PDF, TXT, DOCX, XLSX, XLS, CSV"

        # Add source metadata
        for doc in documents:
            doc.metadata["source"] = filename

        # Split into chunks (unchanged from Day 33)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i

        # Embed and store (unchanged from Day 33)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model
        )

        return (f"Document '{filename}' loaded successfully. "
                f"{len(chunks)} chunks created. "
                f"File type: {suffix.upper()[1:]}")

    finally:
        # Always clean up the temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@tool
def load_text(text: str, source_name: str = "document") -> str:
    """
    Load plain text content into the RAG system.
    Use this when you have the text content directly available.
    For actual files (PDF, DOCX, XLSX, CSV), use load_document instead.
    """
    global vectorstore

    documents = [Document(
        page_content=text,
        metadata={"source": source_name}
    )]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model
    )

    return f"Text loaded as '{source_name}'. {len(chunks)} chunks created."

# =============================================
# SECTION 3: RAG Question Tool (Searches the Document)
# =============================================

@tool
def search_document(question: str) -> str:
    """
    Search the uploaded document to answer a question.
    Use this for questions about documents, reports, policies,
    or any content that has been loaded via load_document.
    Only answers from what's actually in the document.
    """
    if vectorstore is None:
        return "No document loaded yet. Please load a document first using load_document."

    # Search vector store (Day 16-17 logic)
    relevant_chunks = vectorstore.similarity_search(question, k=3)

    context = "\n\n".join([
        f"[Source: {c.metadata.get('source', 'document')}]\n{c.page_content}"
        for c in relevant_chunks
    ])

    # RAG hallucination-prevention prompt (Day 17 pattern, unchanged)
    prompt = f"""You are a helpful AI assistant.
Answer using ONLY the provided context.
If the answer is not in the context, say "I don't have that information in the document."
Always mention which document your answer came from.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=500,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
    )

    return response.text.strip()


# =============================================
# SECTION 4: Web Search Tool (From Day 32)
# =============================================

@tool
def web_search(query: str) -> str:
    """
    Search the web for current, real-time information.
    Use this when asked about recent news, current events,
    live prices, latest developments, or anything that
    requires up-to-date information NOT found in documents.
    """
    results = tavily_client.search(
        query=query,
        max_results=3,
        search_depth="basic"
    )

    formatted = []
    for r in results.get("results", []):
        formatted.append(
            f"Source: {r['url']}\n"
            f"Title: {r['title']}\n"
            f"Content: {r['content'][:300]}..."
        )

    return "\n\n---\n\n".join(formatted) if formatted else "No results found."


# =============================================
# SECTION 5: Agent Setup
# =============================================

api_key = os.environ.get("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0
)

tools = [load_document, load_text, search_document, web_search]
today = date.today().strftime("%B %d, %Y")

agent = create_react_agent(
    llm,
    tools,
    prompt=(
        f"You are a helpful AI research assistant. Today's date is {today}. "
        f"You have THREE tools available:\n"
        f"1. load_document: loads a text document for analysis\n"
        f"2. search_document: searches the loaded document to answer questions\n"
        f"3. web_search: searches the live internet for current information\n\n"
        f"Choose the RIGHT tool for each question:\n"
        f"- Questions about an uploaded document → use search_document\n"
        f"- Questions about current events or live data → use web_search\n"
        f"- Questions needing BOTH sources → use both and synthesize"
    )
)


# =============================================
# SECTION 6: Response Normalization (Unchanged)
# =============================================

def extract_text(content):
    """Normalizes Gemini response content from string or list-of-parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part["text"] for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)


# =============================================
# SECTION 7: Run Agent With Full Trace
# =============================================

def ask_agent(question):
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")

    start_time = time.time()

    result = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })

    elapsed = time.time() - start_time

    for msg in result["messages"]:
        msg_type = type(msg).__name__
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for call in msg.tool_calls:
                print(f"🔧 [{msg_type}] Agent calls: {call['name']}({call['args']})")
        elif msg_type == "ToolMessage":
            content = msg.content[:400] + "..." if len(msg.content) > 400 else msg.content
            print(f"📊 [{msg_type}] Tool result: {content}")
        elif msg_type == "AIMessage" and msg.content:
            clean_text = extract_text(msg.content)
            if clean_text.strip():
                print(f"🤖 [{msg_type}] {clean_text}")

    final_answer = extract_text(result["messages"][-1].content)
    print(f"\n✅ FINAL ANSWER: {final_answer}")
    print(f"⏱️  Agent execution time: {round(elapsed, 2)} seconds")
    time.sleep(15)
    return final_answer