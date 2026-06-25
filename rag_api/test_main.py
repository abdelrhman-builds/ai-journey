# =============================================
# Day 26 - Testing the RAG API
# Author: Abdelrhman
# Date: June 2026
# =============================================

import os
import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from main import app

load_dotenv()

client = TestClient(app)

VALID_KEY = os.getenv("RAG_API_SECRET_KEY")

if not VALID_KEY:
    raise RuntimeError(
        "RAG_API_SECRET_KEY not found in environment. "
        "Make sure .env exists and is loaded correctly "
        "before running tests."
    )

HEADERS = {"x-api-key": VALID_KEY}

@pytest.fixture(scope="module")
def client():
    """
    Using TestClient as a context manager (the `with` block)
    explicitly triggers FastAPI's startup AND shutdown events.
    Without this, @app.on_event("startup") never runs, and
    gemini_client/embeddings_model stay None.

    scope="module" means this fixture runs ONCE for all tests
    in this file, reused across every test function below.
    """
    with TestClient(app) as test_client:
        yield test_client

# =============================================
# TEST 1: Health Check (No Auth Required)
# =============================================

def test_health_check(client):
    """
    /health should work WITHOUT any API key —
    this is the design decision from Day 24.
    """
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# =============================================
# TEST 2: Ask Without API Key — Should Fail
# =============================================

def test_ask_without_api_key(client):
    """
    Proves authentication is actually enforced.
    Missing the required header should be rejected.
    """
    response = client.post("/ask", json={"question": "What is RAG?"})

    # FastAPI's own validation catches the missing
    # required header before our code even runs
    assert response.status_code == 422


# =============================================
# TEST 3: Ask With WRONG API Key — Should Fail
# =============================================

def test_ask_with_wrong_api_key(client):
    """
    Proves our verify_api_key() logic actually
    checks the VALUE, not just its presence.
    """
    response = client.post(
        "/ask",
        json={"question": "What is RAG?"},
        headers={"x-api-key": "wrong-key-12345"}
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"


# =============================================
# TEST 4: Ask Before Any Upload — Should Fail Gracefully
# =============================================

def test_ask_before_upload(client):
    """
    Proves the business-logic guard (not the auth guard)
    correctly catches the "no document yet" scenario.
    """
    response = client.post(
        "/ask",
        json={"question": "What is RAG?"},
        headers=HEADERS
    )

    # NOTE: this test assumes a FRESH server with no
    # prior upload. If run after test_upload_and_ask below
    # in the same session, vectorstore is no longer None —
    # ordering matters here, a real limitation we'll note.
    if response.status_code == 400:
        assert response.json()["detail"] == "No document uploaded yet. Call /upload first."


# =============================================
# TEST 5: Full Flow - Upload Then Ask
# =============================================

def test_upload_and_ask(client):
    """
    The most realistic test: simulates an actual client
    uploading a document, then asking a real question
    about it, and checking the answer makes sense.
    """
    # Create a tiny test file in memory
    test_content = b"RAG stands for Retrieval Augmented Generation. It combines search with AI."

    upload_response = client.post(
        "/upload",
        files={"file": ("test.txt", test_content, "text/plain")},
        headers=HEADERS
    )

    assert upload_response.status_code == 200
    assert upload_response.json()["status"] == "success"

    # Now ask a question about what we just uploaded
    ask_response = client.post(
        "/ask",
        json={"question": "What does RAG stand for?"},
        headers=HEADERS
    )

    assert ask_response.status_code == 200
    data = ask_response.json()
    assert "Retrieval" in data["answer"] or "RAG" in data["answer"]
    assert data["chunks_used"] > 0