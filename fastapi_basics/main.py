# =============================================
# Day 22 - FastAPI Fundamentals
# Author: Abdelrhman
# Date: June 2026
# =============================================

from fastapi import FastAPI

# Create the FastAPI application instance
# This object is the core of everything we build
app = FastAPI(
    title="My First AI API",
    description="Learning FastAPI fundamentals",
    version="1.0.0"
)


# =============================================
# SECTION 1: Your First Endpoint
# =============================================
# @app.get() is a "decorator" — it tells FastAPI:
# "when someone visits this URL with a GET request,
#  run the function below"

@app.get("/")
def read_root():
    """
    Root endpoint - the homepage of your API.
    Visiting http://localhost:8000/ triggers this function.
    """
    return {"message": "Welcome to my first AI API!"}


# =============================================
# SECTION 2: Endpoint With Path Parameters
# =============================================

@app.get("/greet/{name}")
def greet_person(name: str):
    """
    Path parameter example.
    {name} in the URL becomes the 'name' argument here.

    Visiting /greet/Abdelrhman returns a personalized greeting.
    """
    return {"message": f"Hello, {name}! Welcome to AI engineering."}


# =============================================
# SECTION 3: Endpoint With Query Parameters
# =============================================

@app.get("/calculate")
def calculate_percentage(value: float, percentage: float):
    """
    Query parameter example.
    Visiting /calculate?value=100&percentage=15
    automatically fills value=100, percentage=15

    FastAPI automatically converts the URL text into
    float numbers — no manual parsing needed.
    """
    result = (value * percentage) / 100
    return {
        "value": value,
        "percentage": percentage,
        "result": result
    }


# =============================================
# SECTION 4: Optional Query Parameters
# =============================================

@app.get("/study-progress")
def get_study_progress(days_completed: int, total_days: int = 90):
    """
    total_days has a DEFAULT value of 90.
    This makes it OPTIONAL - if not provided, defaults to 90.

    Visiting /study-progress?days_completed=22
    → total_days automatically becomes 90

    Visiting /study-progress?days_completed=22&total_days=100
    → total_days becomes 100 instead
    """
    percentage = (days_completed / total_days) * 100
    return {
        "days_completed": days_completed,
        "total_days": total_days,
        "percentage": round(percentage, 1),
        "days_remaining": total_days - days_completed
    }

# =============================================
# SECTION 5: POST Endpoint - Receiving Data
# =============================================
# POST is used when you're SENDING data to the server,
# not just reading something via the URL.

from pydantic import BaseModel

# Pydantic BaseModel defines the SHAPE of data we expect
# This is what makes FastAPI validate data automatically
class StudyDay(BaseModel):
    day_number: int
    topic: str
    hours_spent: float
    completed: bool


@app.post("/log-study-day")
def log_study_day(day: StudyDay):
    """
    POST endpoint - client sends a JSON body matching StudyDay shape.

    FastAPI automatically:
    1. Parses the incoming JSON
    2. Validates it matches StudyDay (day_number must be int, etc.)
    3. Rejects the request with a clear error if validation fails
    4. Gives us a clean Python object to work with
    """
    return {
        "message": f"Day {day.day_number} logged successfully!",
        "topic": day.topic,
        "hours_spent": day.hours_spent,
        "status": "completed" if day.completed else "in progress"
    }