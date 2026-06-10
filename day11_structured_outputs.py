# =============================================
# Day 11 - Structured Outputs + JSON Extraction
# Author: Abdelrhman
# Date: June 2026
# =============================================
# What we build:
# 1. Job description parser
# 2. CV analyzer
# 3. Data validator
# 4. Batch processor
# =============================================

import os
import json
import time
import google.genai as genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


# =============================================
# HELPER FUNCTIONS
# =============================================

def ask(prompt, system=None, temp=0.0):
    """Gemini API wrapper with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            config = types.GenerateContentConfig(
                temperature=temp,
                max_output_tokens=1000,
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
            if system:
                config.system_instruction = system
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=config
            )
            time.sleep(13)
            return response.text.strip()
        except Exception as e:
            error_str = str(e)
            if ("429" in error_str or "503" in error_str) and attempt < max_retries - 1:
                wait = 60 if "429" in error_str else 30
                print(f"⏳ Waiting {wait}s...")
                time.sleep(wait)
            else:
                return f"Error: {e}"


def parse_json(response_text):
    """
    Safely parse JSON from AI response.
    Handles markdown code fences AI often adds.

    Returns:
        Parsed dictionary or None if parsing fails
    """
    try:
        # Remove markdown fences
        clean = response_text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Raw response: {response_text[:200]}")
        return None


def extract(document, schema, document_type="document"):
    """
    Core extraction function.
    Sends document + schema to AI → returns structured JSON.

    Args:
        document: The text to extract from
        schema: Dictionary showing the desired output structure
        document_type: What kind of document it is (for better prompts)

    Returns:
        Parsed dictionary or None
    """
    # Convert schema dict to formatted JSON string for the prompt
    schema_str = json.dumps(schema, indent=2)

    prompt = f"""Extract information from this {document_type}.
Return ONLY valid JSON matching this exact structure.
No explanation. No markdown. No extra text. Just JSON.

{document_type.upper()}:
{document}

REQUIRED JSON STRUCTURE:
{schema_str}

Rules:
- Use null for missing information
- Use empty list [] if no items found
- Keep values concise
- Extract only what is explicitly stated"""

    response = ask(prompt)
    return parse_json(response)


# =============================================
# SECTION 1: Job Description Parser
# =============================================

print("=" * 60)
print("SECTION 1: Job Description Parser")
print("=" * 60)

# Sample job description
job_description = """
Senior AI Engineer — Cairo, Egypt

We are looking for an experienced AI Engineer to join our
growing team. You will be responsible for building and
deploying machine learning models in production.

Requirements:
- 3+ years of Python experience
- Strong knowledge of machine learning frameworks (TensorFlow, PyTorch)
- Experience with LLMs and prompt engineering
- Familiarity with RAG systems and vector databases
- REST API development with FastAPI or Flask
- Git version control

Nice to have:
- Arabic language skills
- Experience with LangChain or LlamaIndex
- Docker and cloud deployment (AWS/GCP)
- Knowledge of Arabic NLP

Salary: 25,000 - 35,000 EGP per month
Location: Cairo (Hybrid — 3 days office, 2 days remote)
"""
# Define the structure we want extracted
job_schema = {
    "role": "job title",
    "location": "city/country",
    "experience_years": 0,
    "required_skills": ["skill1", "skill2"],
    "nice_to_have": ["skill1", "skill2"],
    "salary_range": {
        "min": 0,
        "max": 0,
        "currency": "currency code"
    },
    "work_type": "remote/hybrid/onsite"
}

print("\nExtracting from job description...")
job_data = extract(job_description, job_schema, "job description")

if job_data:
    print("\n✅ Extracted Job Data:")
    print(json.dumps(job_data, indent=2, ensure_ascii=False))
else:
    print("❌ Extraction failed")


# =============================================
# SECTION 2: CV Analyzer
# =============================================

print("\n" + "=" * 60)
print("SECTION 2: CV Analyzer")
print("=" * 60)

# Sample CV
cv_text = """
Abdelrhman Samir
Cairo, Egypt | abdelrhman@email.com | github.com/abdelrhman-builds

SUMMARY
AI Engineer with background in Financial Auditing.
Building LLM-powered applications with Python and LangChain.
Completed AI & Data Science diploma. Fluent in Arabic and English.

EXPERIENCE
Financial Auditor — ABC Accounting Firm (2021-2023)
- Analyzed financial statements for 50+ clients
- Built Excel automation tools saving 10 hours/week
- Led audit team of 3 junior auditors

EDUCATION
AI & Data Science Diploma — 2026
Bachelor in Business Administration — 2021

SKILLS
Python, Git, LangChain, FastAPI, RAG Systems,
Prompt Engineering, Arabic NLP, Financial Analysis,
Excel, SQL (basic)

LANGUAGES
Arabic (Native), English (Professional)
"""

# CV schema — what we want to extract
cv_schema = {
    "name": "full name",
    "location": "city/country",
    "email": "email address",
    "github": "github url",
    "summary": "2 sentence summary",
    "total_experience_years": 0,
    "skills": ["skill1", "skill2"],
    "languages": ["language1"],
    "education": [
        {
            "degree": "degree name",
            "year": 0
        }
    ],
    "previous_roles": ["role1", "role2"]
}

print("\nExtracting from CV...")
cv_data = extract(cv_text, cv_schema, "CV/resume")

if cv_data:
    print("\n✅ Extracted CV Data:")
    print(json.dumps(cv_data, indent=2, ensure_ascii=False))
else:
    print("❌ Extraction failed")


# =============================================
# SECTION 3: CV-Job Matcher
# =============================================

print("\n" + "=" * 60)
print("SECTION 3: CV-Job Matcher")
print("=" * 60)

def match_cv_to_job(cv_data, job_data):
    """
    Compares extracted CV data against job requirements.
    Returns a match score and analysis.
    """
    if not cv_data or not job_data:
        return None

    # Build matching prompt using extracted data
    prompt = f"""Compare this candidate's CV against the job requirements.
Return ONLY valid JSON. No explanation. No markdown.

CANDIDATE SKILLS: {cv_data.get('skills', [])}
CANDIDATE EXPERIENCE: {cv_data.get('total_experience_years', 0)} years
CANDIDATE LANGUAGES: {cv_data.get('languages', [])}

JOB REQUIRED SKILLS: {job_data.get('required_skills', [])}
JOB NICE TO HAVE: {job_data.get('nice_to_have', [])}
JOB EXPERIENCE NEEDED: {job_data.get('experience_years', 0)} years

Return this exact JSON:
{{
    "match_score": 0,
    "matched_skills": ["skill1"],
    "missing_skills": ["skill1"],
    "matched_nice_to_have": ["skill1"],
    "experience_match": true,
    "recommendation": "one sentence recommendation",
    "strengths": ["strength1", "strength2"],
    "gaps": ["gap1", "gap2"]
}}

Rules:
- match_score: 0-100 percentage
- experience_match: true if candidate meets experience requirement
- Be honest about gaps"""

    response = ask(prompt)
    return parse_json(response)


print("\nMatching CV against job description...")
match_result = match_cv_to_job(cv_data, job_data)

if match_result:
    print("\n✅ Match Analysis:")
    print(json.dumps(match_result, indent=2, ensure_ascii=False))

    # Display summary
    score = match_result.get('match_score', 0)
    print(f"\n{'=' * 40}")
    print(f"MATCH SCORE: {score}/100")
    if score >= 70:
        print("✅ STRONG MATCH — Apply for this job!")
    elif score >= 50:
        print("⚠️  PARTIAL MATCH — Skill up on missing areas")
    else:
        print("❌ WEAK MATCH — Significant gaps to address")
    print(f"{'=' * 40}")
else:
    print("❌ Matching failed")


# =============================================
# SECTION 4: Output Validator
# =============================================

print("\n" + "=" * 60)
print("SECTION 4: Output Validator")
print("=" * 60)


def validate_job_data(data):
    """
    Validates extracted job data has required fields
    and correct data types.

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    if not data:
        return False, ["Data is None"]

    # Check required fields exist
    required_fields = ["role", "location", "required_skills", "work_type"]
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing field: {field}")
        elif data[field] is None:
            errors.append(f"Field is null: {field}")

    # Check data types
    if "required_skills" in data:
        if not isinstance(data["required_skills"], list):
            errors.append("required_skills must be a list")
        elif len(data["required_skills"]) == 0:
            errors.append("required_skills is empty")

    if "experience_years" in data:
        if not isinstance(data["experience_years"], (int, float)):
            errors.append("experience_years must be a number")
        elif data["experience_years"] < 0:
            errors.append("experience_years cannot be negative")

    if "salary_range" in data and data["salary_range"]:
        salary = data["salary_range"]
        if "min" in salary and "max" in salary:
            if salary["min"] > salary["max"]:
                errors.append("salary min cannot be greater than max")

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_cv_data(data):
    """
    Validates extracted CV data has required fields
    and correct data types.
    """
    errors = []

    if not data:
        return False, ["Data is None"]

    # Required fields
    required = ["name", "skills", "education"]
    for field in required:
        if field not in data or data[field] is None:
            errors.append(f"Missing or null field: {field}")

    # Skills must be a non-empty list
    if "skills" in data:
        if not isinstance(data["skills"], list):
            errors.append("skills must be a list")
        elif len(data["skills"]) == 0:
            errors.append("skills list is empty")

    # Experience must be non-negative number
    if "total_experience_years" in data:
        exp = data["total_experience_years"]
        if not isinstance(exp, (int, float)):
            errors.append("total_experience_years must be a number")
        elif exp < 0:
            errors.append("experience cannot be negative")

    is_valid = len(errors) == 0
    return is_valid, errors


# Validate our extracted data
print("\nValidating job data...")
job_valid, job_errors = validate_job_data(job_data)
if job_valid:
    print("✅ Job data is valid")
else:
    print(f"❌ Job data errors: {job_errors}")

print("\nValidating CV data...")
cv_valid, cv_errors = validate_cv_data(cv_data)
if cv_valid:
    print("✅ CV data is valid")
else:
    print(f"❌ CV data errors: {cv_errors}")

# Test validator with intentionally bad data
print("\nTesting validator with bad data...")
bad_data = {
    "role": "Engineer",
    "required_skills": "Python, Git",    # wrong type — string not list
    "experience_years": -1,               # invalid — negative
    "salary_range": {"min": 50000, "max": 30000}  # min > max
}
bad_valid, bad_errors = validate_job_data(bad_data)
print(f"Bad data valid: {bad_valid}")
print(f"Errors found: {bad_errors}")


# =============================================
# SECTION 5: Batch Processor
# =============================================

print("\n" + "=" * 60)
print("SECTION 5: Batch Processor")
print("=" * 60)

# Multiple job descriptions to process
job_listings = [
    {
        "id": "JOB001",
        "text": """
        Python Developer — Alexandria, Egypt
        2+ years Python experience required.
        Skills: Django, PostgreSQL, REST APIs, Git.
        Salary: 15,000-20,000 EGP. Full remote.
        """
    },
    {
        "id": "JOB002",
        "text": """
        Data Scientist — Cairo (On-site)
        5 years experience in ML/AI required.
        Skills: Python, TensorFlow, Pandas, SQL, PowerBI.
        Nice to have: Arabic NLP, Azure ML.
        Salary: 40,000-60,000 EGP monthly.
        """
    },
    {
        "id": "JOB003",
        "text": """
        Junior AI Engineer — Remote
        Fresh graduates welcome. 0-1 years experience.
        Skills: Python, basic ML knowledge, Git.
        Nice to have: LangChain, OpenAI API experience.
        Salary: 8,000-12,000 EGP.
        """
    }
]

# Simple schema for batch processing
# Keep it minimal to save tokens
batch_schema = {
    "role": "job title",
    "location": "city",
    "experience_years": 0,
    "key_skills": ["skill1", "skill2"],
    "salary_min": 0,
    "salary_max": 0,
    "work_type": "remote/hybrid/onsite"
}

print(f"\nProcessing {len(job_listings)} job listings...")
results = []

for i, job in enumerate(job_listings):
    print(f"\nProcessing {job['id']}...")
    extracted = extract(job["text"], batch_schema, "job listing")

    if extracted:
        extracted["id"] = job["id"]    # add ID for tracking
        results.append(extracted)
        print(f"  ✅ {extracted.get('role', 'Unknown')} — {extracted.get('location', 'Unknown')}")
    else:
        print(f"  ❌ Failed to extract {job['id']}")

# Display summary table
print(f"\n{'=' * 60}")
print(f"BATCH PROCESSING COMPLETE — {len(results)}/{len(job_listings)} succeeded")
print(f"{'=' * 60}")
print(f"\n{'ID':<10} {'Role':<25} {'Exp':<6} {'Salary Range':<20} {'Type':<10}")
print("-" * 75)
for r in results:
    role = r.get('role', 'N/A')[:24]
    exp = f"{r.get('experience_years', 0)}yr"
    salary = f"{r.get('salary_min', 0):,}-{r.get('salary_max', 0):,}"
    work = r.get('work_type', 'N/A')
    location = r.get('location') or 'Remote'
    print(f"{r['id']:<10} {role:<25} {exp:<6} {salary:<20} {work:<10}")

# Find best match for our CV
print(f"\n{'=' * 60}")
print("BEST MATCH FOR ABDELRHMAN:")
print(f"{'=' * 60}")

# CV experience = 2 years, skills include Python, FastAPI, LangChain
best_match = None
best_score = 0

for job in results:
    score = 0
    cv_skills = [s.lower() for s in cv_data.get('skills', [])]
    job_skills = [s.lower() for s in job.get('key_skills', [])]

    # Score by skill overlap
    matched = len(set(cv_skills) & set(job_skills))
    score += matched * 20

    # Score by experience match
    exp_required = job.get('experience_years', 0)
    cv_exp = cv_data.get('total_experience_years', 0)
    if cv_exp >= exp_required:
        score += 30

    if score > best_score:
        best_score = score
        best_match = job

if best_match:
    print(f"Role: {best_match.get('role')}")
    print(f"Location: {best_match.get('location') or 'Remote'}")
    print(f"Salary: {best_match.get('salary_min'):,}-{best_match.get('salary_max'):,} EGP")
    print(f"Score: {min(best_score, 100)}/100")