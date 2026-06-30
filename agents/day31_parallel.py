# =============================================
# Day 31 - Parallel Tool Execution 
# Author: Abdelrhman
# Date: June 2026
# =============================================
# Tests whether agent tool calls execute in TRUE parallel
# by making the slow tool genuinely async (await asyncio.sleep)
# instead of blocking (time.sleep) — applying Day 25's lesson.
# =============================================

import os
import time
import asyncio

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


# =============================================
# SECTION 1: Tools (4 from Days 29-30 + the now-async one)
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
    return (f"{skill} in {location}: avg salary {location_data['avg_salary']} EGP, "
            f"demand: {location_data['demand']}, openings: {location_data['openings']}")


@tool
async def slow_market_check(skill: str) -> str:
    """Checks current market trends for a skill. Takes a moment to process."""
    # CHANGED from time.sleep(2) — the blocking version used earlier
    # today produced an AMBIGUOUS 4.8s result. await asyncio.sleep(2)
    # is the NON-BLOCKING version, identical pattern to Day 25's
    # proven slow_operation() demo, where true concurrency was
    # cleanly measurable (6.05s sequential vs 1.99s concurrent).
    await asyncio.sleep(2)
    trends = {
        "Python": "Steady demand, growing 5% yearly",
        "AI Engineer": "Explosive demand, growing 40% yearly",
        "LangChain": "New but rapidly growing, growing 60% yearly"
    }
    return f"{skill} market trend: {trends.get(skill, 'No data available')}"


# =============================================
# SECTION 2: Agent Setup
# =============================================

api_key = os.environ.get("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0
)

tools = [calculate_percentage, calculate_compound_interest, convert_currency,
         search_egypt_tech_jobs, slow_market_check]
agent = create_react_agent(llm, tools)


# =============================================
# SECTION 3: Response Normalization (Unchanged)
# =============================================

def extract_text(content):
    """Normalizes Gemini's response content (string or list-of-parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part["text"] for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)


# =============================================
# SECTION 4: Async Agent Runner With Timing
# =============================================
# NOTE: because one of our tools (slow_market_check) is now
# async, we MUST invoke the agent using the async path,
# agent.ainvoke(), instead of the synchronous agent.invoke()
# used in Days 29-30. This is the exact same .invoke() vs
# await .ainvoke() distinction established in Day 25's RAG
# API work — just applied to the agent framework this time.

async def ask_agent_async(question):
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")

    start_time = time.time()

    result = await agent.ainvoke({
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


