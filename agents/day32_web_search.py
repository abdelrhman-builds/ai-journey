# =============================================
# Day 32 - Web Search Tool Integration (Tavily)
# Author: Abdelrhman
# Date: June 2026
# =============================================
# Adds a real web search tool to the agent, allowing it
# to answer questions about current events and live data
# that hardcoded dictionary tools cannot handle.
# =============================================

import os
import time
from datetime import date

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()


# =============================================
# SECTION 1: Tavily Client Setup
# =============================================
# Initialized ONCE at module level — same principle as
# loading the Gemini client once in the FastAPI startup
# event (Day 23/24). Expensive initialization done once,
# reused for every tool call that follows.

tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))


# =============================================
# SECTION 2: All Tools
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
    """
    Convert an amount between currencies (USD, EUR, GBP, SAR, AED, EGP).
    Note: uses hardcoded approximate rates. For live rates, use web_search instead.
    """
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
def web_search(query: str) -> str:
    """
    Search the web for current, real-time information.
    Use this when asked about recent news, current events,
    live prices, latest developments, or anything that
    requires up-to-date information beyond static knowledge.
    """
    results = tavily_client.search(
        query=query,
        max_results=3,
        search_depth="basic"
    )

    formatted = []
    for r in results.get("results", []):
        formatted.append(
            f"Source: {r['url']}\n"
            f"Title: {r['title']}\n"
            f"Content: {r['content'][:300]}..."
        )

    return "\n\n---\n\n".join(formatted) if formatted else "No results found."


# =============================================
# SECTION 3: Agent Setup
# =============================================

api_key = os.environ.get("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0
)

tools = [
    calculate_percentage,
    calculate_compound_interest,
    convert_currency,
    search_egypt_tech_jobs,
    web_search
]

today = date.today().strftime("%B %d, %Y")  

agent = create_react_agent(
    llm,
    tools,
    prompt=(
        f"You are a helpful AI assistant. "
        f"Today's date is {today}. "
        f"You have access to web_search for real-time information. "
        f"Always use web_search when asked about current events, "
        f"news, live prices, or recent developments — never refuse "
        f"to search based on date assumptions from your training data."
    )
)


# =============================================
# SECTION 4: Response Normalization (Unchanged)
# =============================================

def extract_text(content):
    """Normalizes Gemini response content from either string or list-of-parts format."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part["text"] for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)


# =============================================
# SECTION 5: Run Agent With Full Trace
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
            # Truncate long web search results in the trace
            content = msg.content[:400] + "..." if len(msg.content) > 400 else msg.content
            print(f"📊 [{msg_type}] Tool result: {content}")
        elif msg_type == "AIMessage" and msg.content:
            clean_text = extract_text(msg.content)
            if clean_text.strip():
                print(f"🤖 [{msg_type}] {clean_text}")

    final_answer = extract_text(result["messages"][-1].content)
    print(f"\n✅ FINAL ANSWER: {final_answer}")
    print(f"⏱️  Agent execution time: {round(elapsed, 2)} seconds")
    time.sleep(13)
    return final_answer


# =============================================
# SECTION 6: Test Questions
# =============================================

if __name__ == "__main__":
    # Test 1: Something our hardcoded tools CANNOT answer
    ask_agent("Search for the latest AI developments in 2026 and summarize them.")

    # Test 2: Combining web search WITH existing tools
    ask_agent(
        "Search for the current USD to EGP exchange rate, "
        "then use that rate to convert 500 USD to EGP."
    )