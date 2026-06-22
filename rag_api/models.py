# =============================================
# Day 23 - Pydantic Models for RAG API
# Author: Abdelrhman
# Date: June 2026
# =============================================
# These define the exact SHAPE of every request and
# response our API will handle.

from pydantic import BaseModel
from typing import List


# =============================================
# REQUEST MODELS — what clients send TO us
# =============================================

class QuestionRequest(BaseModel):
    """
    Shape expected when a client asks a question.
    Matches what the Streamlit app sends internally,
    just now formalized as an API contract.
    """
    question: str
    k: int = 3   # how many chunks to retrieve, default 3


# =============================================
# RESPONSE MODELS — what WE send back to clients
# =============================================

class SourceInfo(BaseModel):
    """One retrieved chunk's source information"""
    chunk_id: int
    source_file: str
    preview: str


class AnswerResponse(BaseModel):
    """
    Shape of our answer to a question.
    Defining this means /docs will show clients EXACTLY
    what to expect back, before they even call the API.
    """
    question: str
    answer: str
    sources: List[SourceInfo]
    chunks_used: int


class UploadResponse(BaseModel):
    """Shape of our response after processing an upload"""
    filename: str
    chunks_created: int
    status: str