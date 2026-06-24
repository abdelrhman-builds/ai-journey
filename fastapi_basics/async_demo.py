# =============================================
# Day 25 - Async Demo
# Author: Abdelrhman
# Date: June 2026
# =============================================

import asyncio
import time
from fastapi import FastAPI

app = FastAPI()


# =============================================
# SECTION 1: Simulating a Slow Operation
# =============================================

async def slow_operation(name: str, seconds: int):
    """
    Simulates something slow (like an LLM API call)
    using asyncio.sleep() — the ASYNC version of time.sleep().

    CRITICAL: asyncio.sleep() does NOT block the whole
    program the way time.sleep() does. It tells Python:
    "pause THIS function for N seconds, but feel free to
    run other async code in the meantime."
    """
    print(f"[{name}] Starting, will take {seconds}s...")
    await asyncio.sleep(seconds)
    print(f"[{name}] Finished!")
    return f"{name} result"


# =============================================
# SECTION 2: Synchronous-Style Execution (Sequential)
# =============================================

@app.get("/sequential")
async def run_sequential():
    """
    Runs 3 slow operations ONE AFTER ANOTHER.
    Even though we're inside an async function, awaiting
    them one by one still means waiting for each to
    fully finish before starting the next.
    """
    start = time.time()

    result1 = await slow_operation("Task A", 2)
    result2 = await slow_operation("Task B", 2)
    result3 = await slow_operation("Task C", 2)

    elapsed = time.time() - start
    return {
        "results": [result1, result2, result3],
        "total_time_seconds": round(elapsed, 2)
    }


# =============================================
# SECTION 3: Concurrent Execution (The Real Power)
# =============================================

@app.get("/concurrent")
async def run_concurrent():
    """
    Runs the SAME 3 slow operations, but CONCURRENTLY
    using asyncio.gather(). This starts all 3 at once and
    waits for ALL of them together — not one after another.
    """
    start = time.time()

    result1, result2, result3 = await asyncio.gather(
        slow_operation("Task A", 2),
        slow_operation("Task B", 2),
        slow_operation("Task C", 2)
    )

    elapsed = time.time() - start
    return {
        "results": [result1, result2, result3],
        "total_time_seconds": round(elapsed, 2)
    }