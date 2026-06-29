# =============================================
# Day 29 - LangChain Agents Fundamentals
# Author: Abdelrhman
# Date: June 2026
# =============================================
# Reuses Day 12's tools, but wraps them in a REAL agent
# instead of a manually-written single-step loop.
# =============================================

import os
import time
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()


# =============================================
# SECTION 1: Define Tools (Same Logic as Day 12)
# =============================================
# @tool is LangChain's decorator that turns a normal Python
# function into something an agent can discover and call.
# The function's DOCSTRING becomes the tool's description —
# this is what the agent reads to decide WHEN to use it,
# exactly like Day 12's manual "description=" field did.

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

# =============================================
# SECTION 2: Create the Agent
# =============================================
# create_react_agent() builds the ENTIRE decision loop for us.
# "ReAct" = Reasoning + Acting — the agent alternates between
# THINKING about what to do next and ACTING (calling a tool),
# repeating until it decides it has enough to answer.

api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0
)

tools = [calculate_percentage, calculate_compound_interest, convert_currency, search_egypt_tech_jobs]

agent = create_react_agent(llm, tools)


# =============================================
# SECTION 3: Run the Agent on Test Questions
# =============================================

def ask_agent(question):
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")

    result = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })

    # The agent returns a LIST of all messages exchanged
    # during its reasoning — including every tool call it
    # made along the way. We print each step to SEE the
    # multi-step reasoning happening, not just the final answer.
    for msg in result["messages"]:
        msg_type = type(msg).__name__
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for call in msg.tool_calls:
                print(f"🔧 [{msg_type}] Agent calls: {call['name']}({call['args']})")
        elif msg_type == "ToolMessage":
            print(f"📊 [{msg_type}] Tool result: {msg.content}")
        elif msg_type == "AIMessage" and msg.content:
            print(f"🤖 [{msg_type}] {msg.content}")

    final_answer = result["messages"][-1].content
    print(f"\n✅ FINAL ANSWER: {final_answer}")
    time.sleep(13)
    return final_answer


# =============================================
# SECTION 4: Test Questions — Including Multi-Step Ones
# =============================================

if __name__ == "__main__":
    # Single-step (baseline check — should still work)
    ask_agent("What is the average salary for an AI Engineer in Cairo?")

    # 3-STEP CHAIN: this requires the agent to combine results
    # from job search + currency conversion + percentage calc,
    # in a sequence it must figure out itself
    ask_agent(
        "Find the average AI Engineer salary in Cairo, convert it "
        "to USD, then tell me what 20% of that USD amount would be "
        "(as a hypothetical monthly savings target)."
    )