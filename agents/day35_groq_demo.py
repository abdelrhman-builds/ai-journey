# =============================================
# Day 35 - Groq API Demo (Fast LLM Inference)
# Author: Abdelrhman
# Date: July 2026
# =============================================
# Demonstrates using Groq as a drop-in replacement
# for Gemini in a LangChain agent — same tools,
# same create_react_agent pattern, much faster inference.
# =============================================

import os
import time
from datetime import date
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()


# =============================================
# SECTION 1: Tools (Reused From Days 29-34)
# =============================================

@tool
def calculate_percentage(value: float, percentage: float) -> str:
    """Calculate what percentage of a value is. Use for discounts, tax, tips."""
    result = (value * percentage) / 100
    return f"{percentage}% of {value} = {result}"


@tool
def calculate_compound_interest(principal: float, rate: float, years: float) -> str:
    """Calculate compound interest for savings or loans, given principal, rate (%), and years."""
    amount = principal * (1 + rate / 100) ** years
    interest = amount - principal
    return f"{principal} at {rate}% for {years} years = {round(amount, 2)} (interest earned: {round(interest, 2)})"


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount between currencies (USD, EUR, GBP, SAR, AED, EGP)."""
    rates = {
        "USD": 48.5, "EUR": 52.3, "GBP": 61.2,
        "SAR": 12.9, "AED": 13.2, "EGP": 1.0
    }
    from_rate = rates.get(from_currency.upper(), 1)
    to_rate = rates.get(to_currency.upper(), 1)
    in_egp = amount * from_rate
    result = in_egp / to_rate
    return f"{amount} {from_currency} = {round(result, 2)} {to_currency}"


@tool
def search_egypt_tech_jobs(skill: str, location: str = "Cairo") -> str:
    """Search Egyptian tech job market for salary and demand data for a given skill and location."""
    job_market = {
        "Python": {
            "Cairo": {"avg_salary": 25000, "demand": "High", "openings": 150},
            "Remote": {"avg_salary": 30000, "demand": "High", "openings": 200}
        },
        "AI Engineer": {
            "Cairo": {"avg_salary": 35000, "demand": "Very High", "openings": 80},
            "Remote": {"avg_salary": 45000, "demand": "Very High", "openings": 120}
        },
        "LangChain": {
            "Cairo": {"avg_salary": 40000, "demand": "Very High", "openings": 35},
            "Remote": {"avg_salary": 50000, "demand": "Very High", "openings": 90}
        }
    }
    skill_data = job_market.get(skill, job_market.get("Python"))
    location_data = skill_data.get(location, skill_data.get("Cairo"))
    return (f"{skill} in {location}: avg salary {location_data['avg_salary']} EGP, "
            f"demand: {location_data['demand']}, openings: {location_data['openings']}")


# =============================================
# SECTION 2: Groq LLM Setup
# =============================================
# THE KEY DIFFERENCE FROM DAYS 29-34:
# Only this section changes — everything else is identical.
# LangChain abstracts the provider completely.
#
# Gemini (Days 29-34):
# from langchain_google_genai import ChatGoogleGenerativeAI
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", ...)
#
# Groq (today):
# from langchain_groq import ChatGroq
# llm = ChatGroq(model="llama-3.3-70b-versatile", ...)
#
# Same tools, same agent, same pattern — different provider.
# This proves the value of LangChain's abstraction layer.

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    # Llama 3.3 70B is Meta's latest open-source model,
    # hosted on Groq's LPU hardware for extremely fast inference.
    # Groq's LPUs (Language Processing Units) are purpose-built
    # chips optimized specifically for LLM inference — much faster
    # than GPUs for this specific task.
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0
)

tools = [calculate_percentage, calculate_compound_interest,
         convert_currency, search_egypt_tech_jobs]

today = date.today().strftime("%B %d, %Y")

agent = create_react_agent(
    llm,
    tools,
    prompt=(
        f"You are a helpful AI assistant. Today's date is {today}. "
        f"Use the available tools to answer questions accurately."
    )
)


# =============================================
# SECTION 3: Response Normalization
# =============================================
# Groq returns clean plain strings — no dual-format issue
# like Gemini's list-of-parts. extract_text() still included
# for consistency and defensive coding.

def extract_text(content):
    """Normalizes LLM response content from string or list-of-parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part["text"] for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)


# =============================================
# SECTION 4: Run Agent With Timing
# =============================================

def ask_agent(question):
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")

    start_time = time.time()

    result = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })

    elapsed = time.time() - start_time

    for msg in result["messages"]:
        msg_type = type(msg).__name__
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for call in msg.tool_calls:
                print(f"🔧 [{msg_type}] Agent calls: {call['name']}({call['args']})")
        elif msg_type == "ToolMessage":
            print(f"📊 [{msg_type}] Tool result: {msg.content}")
        elif msg_type == "AIMessage" and msg.content:
            clean_text = extract_text(msg.content)
            if clean_text.strip():
                print(f"🤖 [{msg_type}] {clean_text}")

    final_answer = extract_text(result["messages"][-1].content)
    print(f"\n✅ FINAL ANSWER: {final_answer}")
    print(f"⏱️  Agent execution time: {round(elapsed, 2)} seconds")
    return final_answer


# =============================================
# SECTION 5: Test Questions
# =============================================

if __name__ == "__main__":
    # Single-step test
    ask_agent("What is 15% of 50,000 EGP?")

    # Multi-step test (same as Day 29 — compare speed)
    ask_agent(
        "Find the average AI Engineer salary in Cairo, convert "
        "it to USD, then tell me what 20% of that USD amount is."
    )