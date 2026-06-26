# =============================================
# Day 27 - Pydantic Models with Documentation
# Author: Abdelrhman
# Date: June 2026
# =============================================

from pydantic import BaseModel, Field
from typing import List


# =============================================
# REQUEST MODELS
# =============================================

class QuestionRequest(BaseModel):
    """Request body for asking a question about uploaded documents."""

    question: str = Field(
        ...,
        description="The question to ask about the uploaded document(s)",
        examples=["What is RAG?"]
    )
    k: int = Field(
        default=3,
        description="Number of document chunks to retrieve and use as context",
        ge=1,   # ge = "greater than or equal to" — minimum allowed value
        le=10,  # le = "less than or equal to" — maximum allowed value
        examples=[3]
    )


# =============================================
# RESPONSE MODELS
# =============================================

class SourceInfo(BaseModel):
    """Information about one source chunk used to generate an answer."""

    chunk_id: int = Field(description="Index of this chunk within the document")
    source_file: str = Field(description="Original filename this chunk came from")
    preview: str = Field(description="First ~100 characters of the chunk's text")


class AnswerResponse(BaseModel):
    """Response returned after successfully answering a question."""

    question: str = Field(description="The original question that was asked")
    answer: str = Field(description="AI-generated answer, grounded in the uploaded document")
    sources: List[SourceInfo] = Field(description="Document chunks used to generate the answer")
    chunks_used: int = Field(description="Total number of chunks retrieved for this answer")


class UploadResponse(BaseModel):
    """Response returned after successfully processing an uploaded document."""

    filename: str = Field(description="Name of the file that was uploaded")
    chunks_created: int = Field(description="Number of searchable chunks the document was split into")
    status: str = Field(description="Processing status, e.g. 'success'")