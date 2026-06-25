# =============================================
# Day 26 - Testing FastAPI Apps
# Author: Abdelrhman
# Date: June 2026
# =============================================

from fastapi.testclient import TestClient
from main import app

# TestClient wraps our FastAPI app so we can call its
# endpoints directly in Python, without a running server
client = TestClient(app)


# =============================================
# TEST 1: Root Endpoint
# =============================================

def test_read_root():
    """
    pytest automatically finds functions starting with
    test_ and runs them. Each one should make a request,
    then ASSERT what the result should be.

    `assert` is Python's built-in way of saying:
    "this MUST be true, or the test fails immediately."
    """
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to my first AI API!"}


# =============================================
# TEST 2: Path Parameter Endpoint
# =============================================

def test_greet_person():
    """
    Tests that the path parameter is correctly inserted
    into the response message.
    """
    response = client.get("/greet/Abdelrhman")

    assert response.status_code == 200
    data = response.json()
    assert "Abdelrhman" in data["message"]


# =============================================
# TEST 3: Query Parameters - Correct Math
# =============================================

def test_calculate_percentage():
    """
    Tests that the actual CALCULATION is correct —
    not just that the endpoint responds, but that the
    MATH inside it produces the right answer.
    """
    response = client.get("/calculate?value=100&percentage=15")

    assert response.status_code == 200
    data = response.json()
    assert data["result"] == 15.0


# =============================================
# TEST 4: Optional Parameter Default Value
# =============================================

def test_study_progress_default_total_days():
    """
    Tests that total_days correctly defaults to 90
    when not provided in the request.
    """
    response = client.get("/study-progress?days_completed=22")

    assert response.status_code == 200
    data = response.json()
    assert data["total_days"] == 90
    assert data["days_remaining"] == 68


# =============================================
# TEST 5: POST Request - Valid Data
# =============================================

def test_log_study_day_valid():
    """
    Tests the POST endpoint with CORRECT data —
    should succeed with status 200.
    """
    response = client.post("/log-study-day", json={
        "day_number": 26,
        "topic": "Testing FastAPI",
        "hours_spent": 2.0,
        "completed": True
    })

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"


# =============================================
# TEST 6: POST Request - Invalid Data (Validation)
# =============================================

def test_log_study_day_invalid_type():
    """
    Tests that Pydantic validation correctly REJECTS
    bad data with a 422 error — proving our validation
    safety net actually works, not just assuming it does.
    """
    response = client.post("/log-study-day", json={
        "day_number": "not-a-number",
        "topic": "Testing FastAPI",
        "hours_spent": 2.0,
        "completed": True
    })

    assert response.status_code == 422