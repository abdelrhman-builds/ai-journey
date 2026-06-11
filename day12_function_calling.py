# =============================================
# Day 12 - Function Calling
# Author: Abdelrhman
# Date: June 2026
# =============================================
# What we build:
# 1. Calculator tools
# 2. Egyptian market data tools
# 3. Study tracker tools
# 4. Mini AI assistant using all tools
# =============================================

import os
import json
import time
import math
import google.genai as genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


# =============================================
# SECTION 1: Define Your Python Functions
# =============================================
# These are REAL Python functions the AI can request.
# The AI never runs them directly — YOU run them
# after the AI tells you which one to call and with what args.

def calculate_percentage(value, percentage):
    """Calculate percentage of a value"""
    result = (value * percentage) / 100
    return {
        "value": value,
        "percentage": percentage,
        "result": round(result, 2),
        "explanation": f"{percentage}% of {value} = {result}"
    }

def calculate_compound_interest(principal, rate, years):
    """
    Calculate compound interest.
    Used for savings, loans, investments.
    """
    # A = P(1 + r)^t
    amount = principal * (1 + rate/100) ** years
    interest = amount - principal
    return {
        "principal": principal,
        "rate": rate,
        "years": years,
        "final_amount": round(amount, 2),
        "interest_earned": round(interest, 2),
        "explanation": f"{principal} EGP at {rate}% for {years} years = {round(amount, 2)} EGP"
    }

def convert_currency(amount, from_currency, to_currency):
    """
    Convert between currencies.
    Using approximate rates for Egypt (June 2026).
    """
    # Approximate rates vs EGP
    rates = {
        "USD": 48.5,    # 1 USD = 48.5 EGP
        "EUR": 52.3,    # 1 EUR = 52.3 EGP
        "GBP": 61.2,    # 1 GBP = 61.2 EGP
        "SAR": 12.9,    # 1 SAR = 12.9 EGP
        "AED": 13.2,    # 1 AED = 13.2 EGP
        "EGP": 1.0      # base currency
    }

    from_rate = rates.get(from_currency.upper(), 1)
    to_rate = rates.get(to_currency.upper(), 1)

    # Convert: amount → EGP → target currency
    in_egp = amount * from_rate
    result = in_egp / to_rate

    return {
        "amount": amount,
        "from": from_currency.upper(),
        "to": to_currency.upper(),
        "result": round(result, 2),
        "explanation": f"{amount} {from_currency} = {round(result, 2)} {to_currency}"
    }

def get_study_progress(days_completed, total_days=90):
    """Track AI engineering journey progress"""
    percentage = (days_completed / total_days) * 100
    days_remaining = total_days - days_completed

    # Determine current phase
    if days_completed <= 30:
        phase = "Month 1 — Foundations"
        next_milestone = "Complete RAG chatbot (Day 30)"
    elif days_completed <= 60:
        phase = "Month 2 — AI Engineering Core"
        next_milestone = "Launch freelance profiles (Day 35)"
    else:
        phase = "Month 3 — Advanced + Job Ready"
        next_milestone = "Send 30 job applications (Day 90)"

    return {
        "days_completed": days_completed,
        "days_remaining": days_remaining,
        "percentage": round(percentage, 1),
        "phase": phase,
        "next_milestone": next_milestone,
        "on_track": days_completed >= 10
    }

def search_egypt_tech_jobs(skill, location="Cairo"):
    """
    Simulates searching Egyptian tech job market.
    In production this would call a real jobs API.
    """
    # Simulated job market data
    job_market = {
        "Python": {
            "Cairo": {"avg_salary": 25000, "demand": "High", "openings": 150},
            "Alexandria": {"avg_salary": 18000, "demand": "Medium", "openings": 45},
            "Remote": {"avg_salary": 30000, "demand": "High", "openings": 200}
        },
        "AI Engineer": {
            "Cairo": {"avg_salary": 35000, "demand": "Very High", "openings": 80},
            "Alexandria": {"avg_salary": 25000, "demand": "Medium", "openings": 20},
            "Remote": {"avg_salary": 45000, "demand": "Very High", "openings": 120}
        },
        "LangChain": {
            "Cairo": {"avg_salary": 40000, "demand": "Very High", "openings": 35},
            "Remote": {"avg_salary": 50000, "demand": "Very High", "openings": 90}
        }
    }

    skill_data = job_market.get(skill, job_market.get("Python"))
    location_data = skill_data.get(location, skill_data.get("Cairo"))

    return {
        "skill": skill,
        "location": location,
        "avg_salary_egp": location_data["avg_salary"],
        "demand": location_data["demand"],
        "estimated_openings": location_data["openings"],
        "recommendation": f"{skill} in {location} has {location_data['demand']} demand"
    }


# =============================================
# SECTION 2: Tool Definitions for Gemini
# =============================================
# We must tell Gemini WHAT tools exist and HOW to call them.
# This is written as a list of function declarations.
# Gemini reads these and decides which to call.

tools = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="calculate_percentage",
                description="Calculate what percentage of a value is. Use for discounts, tax calculations, tips.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "value": types.Schema(
                            type=types.Type.NUMBER,
                            description="The base value"
                        ),
                        "percentage": types.Schema(
                            type=types.Type.NUMBER,
                            description="The percentage to calculate"
                        )
                    },
                    required=["value", "percentage"]
                )
            ),
            types.FunctionDeclaration(
                name="calculate_compound_interest",
                description="Calculate compound interest for savings or loans.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "principal": types.Schema(
                            type=types.Type.NUMBER,
                            description="Initial amount in EGP"
                        ),
                        "rate": types.Schema(
                            type=types.Type.NUMBER,
                            description="Annual interest rate as percentage"
                        ),
                        "years": types.Schema(
                            type=types.Type.NUMBER,
                            description="Number of years"
                        )
                    },
                    required=["principal", "rate", "years"]
                )
            ),
            types.FunctionDeclaration(
                name="convert_currency",
                description="Convert between currencies including EGP, USD, EUR, GBP, SAR, AED.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "amount": types.Schema(
                            type=types.Type.NUMBER,
                            description="Amount to convert"
                        ),
                        "from_currency": types.Schema(
                            type=types.Type.STRING,
                            description="Source currency code (USD, EUR, EGP, etc)"
                        ),
                        "to_currency": types.Schema(
                            type=types.Type.STRING,
                            description="Target currency code"
                        )
                    },
                    required=["amount", "from_currency", "to_currency"]
                )
            ),
            types.FunctionDeclaration(
                name="get_study_progress",
                description="Get current progress in the 90-day AI engineering study plan.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "days_completed": types.Schema(
                            type=types.Type.NUMBER,
                            description="Number of days completed so far"
                        )
                    },
                    required=["days_completed"]
                )
            ),
            types.FunctionDeclaration(
                name="search_egypt_tech_jobs",
                description="Search Egyptian tech job market for salary and demand data.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "skill": types.Schema(
                            type=types.Type.STRING,
                            description="Technical skill to search for"
                        ),
                        "location": types.Schema(
                            type=types.Type.STRING,
                            description="City in Egypt or Remote"
                        )
                    },
                    required=["skill"]
                )
            )
        ]
    )
]


# =============================================
# SECTION 3: Function Router
# =============================================
# When AI requests a function call, we need to
# route it to the correct Python function.

def run_function(name, args):
    """
    Routes AI function call requests to real Python functions.

    Args:
        name: Function name AI wants to call
        args: Arguments AI wants to pass

    Returns:
        Function result as dictionary
    """
    function_map = {
        "calculate_percentage": calculate_percentage,
        "calculate_compound_interest": calculate_compound_interest,
        "convert_currency": convert_currency,
        "get_study_progress": get_study_progress,
        "search_egypt_tech_jobs": search_egypt_tech_jobs
    }

    if name in function_map:
        return function_map[name](**args)
    else:
        return {"error": f"Unknown function: {name}"}


# =============================================
# SECTION 4: AI Assistant with Tools
# =============================================

def ask_with_tools(question):
    """
    Sends question to AI with tools available.
    Handles the full function calling loop:
    1. Send question + tools to AI
    2. If AI wants to call a function → run it
    3. Send result back to AI
    4. Get final answer
    """
    print(f"\n{'='*50}")
    print(f"Question: {question}")
    print(f"{'='*50}")

    try:
    # Step 1: Send question with tools
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=question,
            config=types.GenerateContentConfig(
                tools=tools,
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )

        # Step 2: Check if AI wants to call a function
        if response.candidates[0].content.parts[0].function_call:
            function_call = response.candidates[0].content.parts[0].function_call
            func_name = function_call.name
            func_args = dict(function_call.args)

            print(f"🔧 AI calls: {func_name}({func_args})")

            # Step 3: Run the actual Python function
            func_result = run_function(func_name, func_args)
            print(f"📊 Result: {func_result}")

            # Step 4: Send result back to AI for final answer
            final_response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    {"role": "user", "parts": [{"text": question}]},
                    {"role": "model", "parts": [{"function_call": {
                        "name": func_name,
                        "args": func_args
                    }}]},
                    {"role": "user", "parts": [{"function_response": {
                        "name": func_name,
                        "response": func_result
                    }}]}
                ],
                config=types.GenerateContentConfig(
                    tools=tools,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )
            )
            answer = final_response.text.strip()
        else:
            # AI answered directly without tools
            answer = response.text.strip()

        print(f"🤖 Answer: {answer}")
        time.sleep(13)
        return answer

    except Exception as e:              # ← catch errors gracefully
            print(f"⚠️ Error: {e}")
            return None

# =============================================
# SECTION 5: Test the Assistant
# =============================================

print("=" * 60)
print("AI ASSISTANT WITH FUNCTION CALLING")
print("=" * 60)

questions = [
    "What is 15% of 50,000 EGP?",
    "If I save 10,000 EGP at 18% interest for 3 years, how much will I have?",
    "How much is 500 USD in Egyptian Pounds?",
    "What is my progress after completing 12 days of my 90-day AI study plan?",
    "What is the job market like for AI Engineers in Cairo?"
]

# =============================================
# SECTION 6: Interactive Assistant
# =============================================

def run_assistant():
    """
    Interactive assistant that uses tools to answer questions.
    Type 'quit' to exit.
    """
    system_prompt = """You are Zaki, a helpful Egyptian AI assistant.
You have access to financial calculation tools and job market data.
Always use the available tools when asked for calculations or job data.
Respond in a friendly, helpful way. Keep answers concise."""

    print("\n" + "=" * 50)
    print("   🤖 Zaki — Egyptian AI Financial Assistant")
    print("=" * 50)
    print("Ask me about calculations, currency, or jobs!")
    print("Type 'quit' to exit")
    print("-" * 50)

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        if user_input.lower() == 'quit':
            print("Zaki: Goodbye! Keep building! 🚀")
            break

        # Send to AI with tools
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_input,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=tools,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )
            )

            # Check for function call
            part = response.candidates[0].content.parts[0]

            if hasattr(part, 'function_call') and part.function_call:
                func_name = part.function_call.name
                func_args = dict(part.function_call.args)

                # Run function silently
                func_result = run_function(func_name, func_args)

                # Get final answer
                final = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        {"role": "user", "parts": [{"text": user_input}]},
                        {"role": "model", "parts": [{"function_call": {
                            "name": func_name,
                            "args": func_args
                        }}]},
                        {"role": "user", "parts": [{"function_response": {
                            "name": func_name,
                            "response": func_result
                        }}]}
                    ],
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        tools=tools,
                        thinking_config=types.ThinkingConfig(thinking_budget=0)
                    )
                )
                print(f"\nZaki: {final.text.strip()}")
            else:
                print(f"\nZaki: {response.text.strip()}")

            time.sleep(13)

        except Exception as e:
            print(f"\nZaki: Sorry, I encountered an error: {e}")


# =============================================
# RUN EVERYTHING
# =============================================

if __name__ == "__main__":

    # First: automated tests (uses quota)
    print("=" * 60)
    print("AI ASSISTANT WITH FUNCTION CALLING")
    print("=" * 60)

    for question in questions:
        ask_with_tools(question)
        print()

    # Then: interactive mode
    print("\n" + "=" * 50)
    print("Starting interactive mode...")
    run_assistant()