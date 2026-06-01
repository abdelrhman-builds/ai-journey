# =============================================
# Day 2 - Python Fundamentals
# Author: Abdelrhman
# Date: June 2026
# =============================================

# --- SECTION 1: Variables and Data types ---

name = "Abdelrhman"
age = 20
height = 1.75
is_student = True

print("Name:", name)
print("Age:", age)
print("Height:", height)
print("Is_student:", is_student)
print("Type of name:", type(name))
print("Type of age:", type(age))
print("Type of height:", type(height))
print("Type of is_student:", type(is_student))


# --- SECTION 2: String Operations ---

full_name = "abdelrhman samir"
course = "AI Engineering"
days = 90

# String methods
print(full_name.upper())
print(full_name.capitalize())  
print(full_name.title())       
print(len(full_name))
print(full_name.replace("abdelrhman", "Abdelrhman"))

# F-string
print(f"My name is {full_name.capitalize()} and I am studying {course} for {days} days")

# Split and join
words = full_name.split()
print(words)
print(" ".join(words))

sentence = "  machine learning is amazing  "
words = sentence.strip().title().split()
print(words)


# --- SECTION 3: Lists ---

# your AI learning Journey as a list
skills = ["Python", "Git", "OpenAI API", "RAG", "FastAPI"]

# Access items
print("First skill:", skills[0])
print("Last skill:", skills[-1])

# Add a new skill
skills.append("LangChain")
print("After adding Langchain:", skills)

# Remove a skill
skills.remove("Git")
print("After removing Git:", skills)

# Check length
print("Total skills:", len(skills))

# Check if skill exists
print("Is Python in skills?", "Python" in skills)
print("Is Docker in skills?", "Docker" in skills)

# Slice -- first 3 skills
print("First 3 skills:", skills[:3])


# --- SECTION 4: Dictionaries ---

# Your AI engineer profile
profile = {
    "name": "Abdelrhman",
    "city": "Cairo",
    "course": "AI Engineering",
    "days_completed": 2,
    "skills": ["Python", "Git", "GitHub"]
}

# Access values
print("Name:", profile["name"])
print("City:", profile["city"])

# Safe access 
print("Salary:", profile.get("salary", "Not specified"))

# Add new key 
profile["goal"] = "junior AI Engineer"
print("Goal added:", profile["goal"])

# Update existing key
profile["days_completed"] = 2
print("Days completed:", profile["days_completed"])

# Check keys
print("Is 'name' in profile?", "name" in profile)
print("Is 'age' in profile?", "age" in profile)

# Print all keys
print("All keys:", list(profile.keys()))

# Access the skills list inside the dictionary
print("First skill:", profile["skills"][0])


# --- SECTION 5: Loops ---

ai_topics = ["Python", "LLM", "RAG", "Agents", "Fine-tuning"]

# Loop 1 - simple for loop
print("=== AI Topics ===")
for topic in ai_topics:
    print(topic)

# loop 2 - with enumerate
print("\n=== Numbered List ===")
for index, topic in enumerate(ai_topics):
    print(f"{index + 1}. {topic}")

# Loop 3 — loop through dictionary
print("\n=== My Profile ===")
profile = {"name": "Abdelrhman", "city": "Cairo", "course": "AI Engineering"}
for key, value in profile.items():
    print(f"{key}: {value}")

# Loop 4 — loop with condition
print("\n=== Score Checker ===")
scores = [95, 42, 87, 31, 96, 55]
for score in scores:
    if score >= 60:
        print(f"{score} → Pass ✅")
    else:
        print(f"{score} → Fail ❌") 


# --- SECTION 6: Functions ---

# Function 1 — Greet a user
def greet_user(name, role="AI Engineer"):
    return f"Welcome, {name}! You are training to become a {role}"

# Function 2 — Calculate study progress
def calculate_progress(days_completed, total_days=90):
    percentage = (days_completed / total_days) * 100
    return f"Progress: {percentage:.1f}% ({days_completed}/{total_days} days)"

# Function 3 — Clean text
def clean_text(text):
    return text.strip().lower()

# Function 4 — Check if score passes
def check_score(score, passing_grade=60):
    if score >= passing_grade:
        return f"{score} → Pass ✅"
    else:
        return f"{score} → Fail ❌"
    
# Function 5 — Build a simple prompt
def build_prompt(topic, level="beginner"):
    return f"Explain {topic} in simple terms for a {level} student."

# --- Call all functions ---
print(greet_user("Abdelrhman"))
print(greet_user("Abdelrhman", "RAG Specialist"))
print(calculate_progress(2))
print(calculate_progress(45))
print(clean_text("  Machine Learning  "))
print(check_score(95))
print(check_score(42))
print(build_prompt("RAG systems"))
print(build_prompt("transformers", "intermediate"))


# --- SECTION 7: Putting It All Together ---
def analyze_skills(skills):
    count = len(skills)
    for index, skill in enumerate(skills):
        print(f"{index + 1}. {skill}")
    return {
        "total": count,
        "first": skills[0],
        "last": skills[-1]
    }

my_skills = ["Python", "Git", "OpenAI API", "RAG", "FastAPI"]
result = analyze_skills(my_skills)
print(f"Total skills: {result['total']}")
print(f"First skill: {result['first']}")
print(f"Last skill: {result['last']}")