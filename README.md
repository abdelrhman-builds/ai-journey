# 🤖 AI Journey — 90-Day AI Engineering Plan

> Building from Python fundamentals to deployed AI applications.
> Follow my progress as I ship real projects every week.

---

## 👨‍💻 About Me
AI Engineer based in Cairo, Egypt.
Building LLM-powered applications with Python, LangChain, and RAG systems.
---

## 🚀 Projects

### 🌟 Featured Project: AI Document Q&A System (RAG)
**[🔗 Live Demo](https://huggingface.co/spaces/AbdelrhmanSamir/ai-document-qa)**

A production RAG (Retrieval-Augmented Generation) system that answers
questions from uploaded documents with zero hallucination and source citations.

![Status](https://img.shields.io/badge/Status-Live-success)
![Deployed](https://img.shields.io/badge/Deployed-HuggingFace%20Spaces-orange)

![Upload Interface](assets/rag-demo-upload.png)
![Q&A with Sources](assets/rag-demo-answer.png)

**Features:**
- Multi-format document upload (PDF, TXT)
- Semantic search with local embeddings (HuggingFace)
- Zero-hallucination answers with source citations
- Multi-document support with metadata filtering
- Production security: file validation, usage limits, quota protection
- Clean chat-based UI built with Streamlit

**Stack:** Python, LangChain, ChromaDB, HuggingFace Embeddings, Google Gemini API, Streamlit, Docker

**Try it:** Upload any PDF and ask questions — the system only answers
from what's actually in your document, and tells you when it doesn't know.

### 🔌 RAG API Backend (FastAPI)
**[🔗 Live API Docs](https://abdelrhmansamir-rag-api-backend.hf.space/docs)**
 
A standalone REST API exposing the same RAG system as a documented,
authenticated, asynchronous backend — separate from the Streamlit UI
above. Built for integration into other applications, not just human
browser use.
 
![Status](https://img.shields.io/badge/Status-Live-success)
![Tested](https://img.shields.io/badge/Tests-Passing-brightgreen)
![Deployed](https://img.shields.io/badge/Deployed-HuggingFace%20Spaces-orange)
 
**Source code:** [`rag_api/`](./rag_api) in this repository
**Deployment:** Docker container on HuggingFace Spaces
*(Note: the HuggingFace Space's own git repo is a deployment target
only — this repository's `rag_api/` folder is the canonical source.)*
 
**Features:**
- API key authentication (`X-API-Key` header)
- Async request handling for concurrent clients
- Automated test suite (pytest + TestClient) — 11 tests passing
- Full interactive documentation with field-level descriptions and validation

**Stack:** FastAPI, Pydantic, pytest, Docker, async/await, ChromaDB,
HuggingFace Embeddings, Google Gemini API
 
---
 
**Example usage:**
 
Upload a document:
```bash
curl -X POST "https://abdelrhmansamir-rag-api-backend.hf.space/upload" \
  -H "x-api-key: YOUR_API_KEY" \
  -F "file=@sample_document.txt"
```
 
Response:
```json
{
  "filename": "sample_document.txt",
  "chunks_created": 8,
  "status": "success"
}
```
 
Ask a question:
```bash
curl -X POST "https://abdelrhmansamir-rag-api-backend.hf.space/ask" \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "k": 3}'
```
 
Response:
```json
{
  "question": "What is RAG?",
  "answer": "RAG stands for Retrieval Augmented Generation. It is a technique that combines document retrieval with language model generation to produce accurate, grounded answers. (Source: sample_document.txt)",
  "sources": [
    {
      "chunk_id": 1,
      "source_file": "sample_document.txt",
      "preview": "Key responsibilities of an AI Engineer:\n- Building LLM-powered applications\n- Implementing RAG syste"
    }
  ],
  "chunks_used": 3
}
```
 
*Note: this is a demo with limited free-tier API quota. For full access
or a live walkthrough, feel free to reach out via LinkedIn.*

### 🤖 AI Agents (Week 5)
**Source code:** [`agents/`](./agents) in this repository

A progression of increasingly sophisticated AI agents built with
LangChain and LangGraph:

| Day | File | What It Demonstrates |
|-----|------|----------------------|
| 29 | `day29_basic_agent.py` | First real agent — multi-step tool chaining with `create_react_agent` |
| 30 | `day30_chaining.py` | 3-step sequential chain — job search → currency convert → percentage |
| 31 | `day31_parallel.py` | Parallel tool execution investigation — timing analysis with async tools |
| 32 | `day32_web_search.py` | Live web search via Tavily + dynamic date system prompt |
| 33 | `day33_rag_agent.py` | RAG + web search combined — agent routes between private documents and live internet |
| 34 | `day34_rag_agent.py` | Multi-format file support (PDF, DOCX, CSV, XLSX) added to the RAG agent |
| 35 | `day35_groq_demo.py` | **Groq provider swap** — same agent, Llama 3.3 70B, ~5-6x faster than Gemini |

**Tools built across Week 5:**
- Financial calculators (percentage, compound interest, currency)
- Egyptian tech job market lookup
- Live web search (Tavily)
- RAG document loader and searcher (multi-format)

**Key capability proven on Day 33:**
One agent answering from both your uploaded documents AND the live web,
choosing the right source automatically and synthesizing both when needed.

---

### 📝 NLP Fundamentals (Week 6)
**Source code:** [`nlp/`](./nlp) in this repository

A progression from how text becomes numbers to real, working Arabic-aware
NLP systems:

| Day | File | What It Demonstrates |
|-----|------|----------------------|
| 36 | `day36_tokenization.py` | Tokenization — NLTK, BERT WordPiece, GPT-2 BPE compared side by side |
| 37 | `day37_embeddings.py` | TF-IDF + Word2Vec — statistical vs neural word representations |
| 38 | `day38_transformers.py` | Attention mechanism — real BERT attention weights extracted and visualized |
| 39 | `day39_pipelines.py` | HuggingFace pipelines — sentiment, NER, zero-shot classification, summarization |
| 40 | `day40_arabic_nlp.py` | **AraBERT vs generic multilingual model** — proven confidence gap on real Egyptian reviews |
| 41 | `day41_finetuning.py` | Fine-tuning — custom 3-category classifier trained on just 9 examples |
| 42 | `feedback_analyzer.py` | **Capstone** — bilingual customer feedback analyzer (language detection, sentiment routing, NER, categorization) |

**Key finding proven on Day 40:**
AraBERT (Arabic-specialized) was consistently far more confident than a
generic multilingual model on real Egyptian business reviews — 96.5% vs
59.1%, 91.2% vs 41.7%, 98% vs 42% — a repeatable, measurable improvement.

---

### 👁️ Computer Vision (Week 7)
**Source code:** [`computer_vision/`](./computer_vision) in this repository

From how images become numbers to a production-grade bilingual vision tool:

| Day | File | What It Demonstrates |
|-----|------|----------------------|
| 43 | `day43_cnn.py` | CNN built from scratch — 98.54% test accuracy on MNIST |
| 44 | `day44_transfer_learning.py` | Transfer learning with ResNet-18 — only 0.05% of parameters trained |
| 45 | `day45_yolo.py` | Object detection with YOLO — multi-object, real-time speed |
| 46 | `day46_segmentation.py` | Instance segmentation — pixel-precise masks vs bounding boxes |
| 47 | `day47_ocr.py` | Bilingual OCR (English + Arabic) — output ready for RAG ingestion |
| 48 | `capstone/document_analyzer.py` | **Most advanced** — production-grade bilingual (EN/AR) image analyzer with full error handling |

**Key findings proven with real numbers:**
- ResNet-18 transfer learning: only 5,130 of 11,181,642 parameters (0.05%) needed training, and accuracy scaled correctly with more data (69% → 78.87%)
- A bus's bounding box was only 22.2% actual object — 77.8% was background
- Bilingual OCR correctly extracted both English and Arabic text from real images

---

### Project 0: CSV Analyzer (Day 5)
A command-line tool that analyzes any CSV file and displays statistics.
- **Stack:** Python, argparse, csv module
- **Run:** `cd csv_analyzer && python analyzer.py --file data.csv`

### Project 1: Zaki — AI Chatbot (Day 9)
Multi-turn terminal chatbot with memory and token tracking.
- **Stack:** Python, Gemini API, google-genai
- **Run:** `cd chatbot && python day9_chatbot.py`

### Project 2: Prompt Engineering Lab (Day 10)
Experiments with zero-shot, few-shot, CoT, and templates.
- **Stack:** Python, Gemini API, PromptTemplate class
- **Run:** `python day10_prompts.py`

### Project 3: AI Data Extractor (Day 11)
Extracts structured JSON from job descriptions and CVs.
- **Stack:** Python, Gemini API, JSON validation
- **Run:** `python day11_structured_outputs.py`

### Project 4: Zaki Financial Assistant (Day 12)
AI assistant with function calling — calculator, currency, jobs.
- **Stack:** Python, Gemini API, function calling
- **Run:** `python day12_function_calling.py`

---

## 📅 Progress Tracker

| Day | Topic | Status |
|-----|-------|--------|
| 1 | Environment Setup | ✅ |
| 2 | Python Fundamentals | ✅ |
| 3 | File Handling + Git | ✅ |
| 4 | OOP | ✅ |
| 5 | CSV Analyzer Project | ✅ |
| 6-7 | Git Mastery + Review | ✅ |
| 8 | Gemini API Basics | ✅ |
| 9 | Multi-turn Chatbot | ✅ |
| 10 | Prompt Engineering | ✅ |
| 11 | Structured Outputs | ✅ |
| 12 | Function Calling | ✅ |
| 13-14 | LinkedIn + Week 2 Review | ✅ |
| 15 | RAG Foundations | ✅ |
| 16 | Embeddings + Vector Store | ✅ |
| 17 | Full RAG Pipeline | ✅ |
| 18 | Multi-Document RAG | ✅ |
| 19 | Streamlit UI | ✅ |
| 20 | Permanent Deployment | ✅ |
| 21 | Week 3 Wrap-up | ✅ |
| 22 | FastAPI Fundamentals | ✅ |
| 23 | FastAPI + RAG Integration | ✅ |
| 24 | API Authentication | ✅ |
| 25 | Async Patterns | ✅ |
| 26 | Automated Testing | ✅ |
| 27 | API Documentation Polish | ✅ |
| 28 | API Deployment | ✅ |
| 29 | LangChain Agents Fundamentals | ✅ |
| 30 | Sequential Tool Chaining | ✅ |
| 31 | Parallel Tool Execution | ✅ |
| 32 | Web Search Integration | ✅ |
| 33 | Agent + RAG Combined | ✅ |
| 34 | Multi-Format RAG Agent | ✅ |
| 35 | Week 5 Wrap-up + Groq Demo | ✅ |
| 36 | Tokenization | ✅ |
| 37 | TF-IDF + Word2Vec | ✅ |
| 38 | Transformer Attention | ✅ |
| 39 | HuggingFace Pipelines | ✅ |
| 40 | Arabic NLP + AraBERT | ✅ |
| 41 | Fine-tuning | ✅ |
| 42 | Week 6 Wrap-up + Capstone | ✅ |
| 43 | CNNs | ✅ |
| 44 | Transfer Learning | ✅ |
| 45 | Object Detection (YOLO) | ✅ |
| 46 | Instance Segmentation | ✅ |
| 47 | OCR (English + Arabic) | ✅ |
| 48 | Computer Vision Capstone | ✅ |
| 49 | Week 7 Wrap-up | ✅ |
| 50+ | Production Engineering (SQL, Docker, CI/CD) | ⏳ |

---

## 🛠️ Tech Stack
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Gemini](https://img.shields.io/badge/Gemini-API-orange)
![Git](https://img.shields.io/badge/Git-2.54-red)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![LangChain](https://img.shields.io/badge/LangChain-Agents-green)

---

## 📫 Connect
- GitHub: [@abdelrhman-builds](https://github.com/abdelrhman-builds)
- LinkedIn: [Abdelrhman Samir](https://www.linkedin.com/in/abdelrhman-samir-ai)
- Journey: Building AI applications one day at a time