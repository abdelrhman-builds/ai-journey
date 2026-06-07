# =============================================
# Day 8 - Gemini API Basics
# Author: Abdelrhman
# Date: June 2026
# =============================================

import os                          # built-in module to access environment variables
import google.genai as genai       # Google's new Gemini SDK (google-genai package)
from google.genai import types     # types module for configuration objects
from dotenv import load_dotenv     # loads variables from .env file into environment

# =============================================
# SETUP: Load API Key
# =============================================

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GEMINI_API_KEY")

# Verify key loaded
if api_key:
    print("✅ Environment setup correct!")
else:
    print("❌ API key not found! Check your .env file")

# Initialize client
client = genai.Client(api_key=api_key)

# =============================================
# SECTION 1: First AI Call
# =============================================
print("\n=== First AI Call ===")
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain what RAG is in 3 simple sentences."
)
print(response.text)


# =============================================
# SECTION 2: Understanding the Response Object
# =============================================

response2 = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is a vector database? Answer in 2 sentences."
)

# The response has multiple parts
print("\n=== Response Text ===")
print(response2.text)

print("\n=== Usage Metadata ===")
print(response2.usage_metadata)


# =============================================
# SECTION 3: Temperature Experiments 
# =============================================

prompt = "Give me a one-sentence tagline for an AI engineering course."

print("\n=== Temperature Experiments ===")

temperatures = [0.0, 1.0, 2.0]

for temp in temperatures:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=200,    # increased from 50
            thinking_config=types.ThinkingConfig(
                thinking_budget=0     # disable thinking to save tokens
            )
        )
    )
    print(f"\nTemperature {temp}:")
    print(f"  {response.text.strip()}")

    
# =============================================
# SECTION 4: System Prompts
# =============================================
from google.genai import types

print("\n=== System Prompt Experiments ===")

# Experiment 1 — Without system prompt
response1 = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is Python?",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        max_output_tokens=100
    )
)
print("Without system prompt:")
print(response1.text.strip())

# Experiment 2 — With system prompt (Cairo tutor)
response2 = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is Python?",
    config=types.GenerateContentConfig(
        system_instruction="You are a Python tutor teaching a student in Cairo, Egypt. Always use simple Arabic-friendly examples. Keep answers under 50 words.",
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        max_output_tokens=100
    )
)
print("\nWith system prompt (Cairo tutor):")
print(response2.text.strip())

# Experiment 3 — System prompt for RAG
response3 = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is the capital of France?",
    config=types.GenerateContentConfig(
        system_instruction="You are a helpful assistant. Answer ONLY using the provided context. If the answer is not in the context, say 'I don't know'. Context: Python is a programming language created in 1991.",
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        max_output_tokens=100
    )
)
print("\nRAG-style system prompt (should say I don't know):")
print(response3.text.strip())


# =============================================
# SECTION 5: Reusable AI Function
# =============================================

def ask_ai(question, system_prompt=None, temperature=0.7, max_tokens=500):
    """
    Reusable function to call Gemini API.

    Args:
        question: The question to ask
        system_prompt: Optional instructions for the AI
        temperature: Creativity level (0.0 to 2.0)
        max_tokens: Maximum response length

    Returns:
        AI response as string, or error message
    """
    try:
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )

        # Add system prompt if provided
        if system_prompt:
            config.system_instruction = system_prompt

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=question,
            config=config
        )
        return response.text.strip()

    except Exception as e:
        return f"Error: {e}"


# --- Test the function ---
print("\n=== Testing ask_ai Function ===")

# Simple question
print("\n1. Simple question:")
print(ask_ai("What is machine learning in one sentence?"))

# With system prompt
print("\n2. With system prompt:")
print(ask_ai(
    question="What is machine learning?",
    system_prompt="You are a teacher. Explain everything using food analogies.",
    temperature=0.7
))

# Factual — low temperature
print("\n3. Factual (low temperature):")
print(ask_ai(
    question="What year was Python created?",
    temperature=0.0
))

# Creative — high temperature
print("\n4. Creative (high temperature):")
print(ask_ai(
    question="Write a one-line poem about coding.",
    temperature=1.5
))