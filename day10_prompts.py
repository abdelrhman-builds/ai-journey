# =============================================
# Day 10 - Prompt Engineering Deep Dive
# Author: Abdelrhman
# Date: June 2026
# =============================================

import time
import os
import google.genai as genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# Reusable ask function from Day 8
def ask(prompt, system=None, temp=0.0):
    """Simple wrapper for Gemini API calls with retry"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            config = types.GenerateContentConfig(
                temperature=temp,
                max_output_tokens=500,
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
            if system:
                config.system_instruction = system
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=config
            )
            return response.text.strip()
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait_time = 60   # wait 60 seconds then retry
                print(f"\n  ⏳ Rate limit hit — waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                return f"Error: {e}"


# =============================================
# SECTION 1: Zero-Shot vs Few-Shot
# =============================================

print("=" * 60)
print("SECTION 1: Zero-Shot vs Few-Shot Prompting")
print("=" * 60)

# --- Zero-Shot ---
zero_shot_prompt = """Classify the sentiment of this text.
Text: "I just deployed my first AI application!"
Sentiment:"""

print("\n[Zero-Shot]")
print(f"Prompt: {zero_shot_prompt}")
print(f"Response: {ask(zero_shot_prompt)}")

# --- Few-Shot ---
few_shot_prompt = """Classify the sentiment of texts. Output only: Positive, Negative, or Neutral.

Examples:
Text: "I love building AI apps!" → Positive
Text: "This bug is so frustrating" → Negative
Text: "The model returned a response" → Neutral
Text: "My code finally works after 3 hours!" → Positive
Text: "I failed the interview" → Negative

Now classify:
Text: "I just deployed my first AI application!"
Sentiment:"""

print("\n[Few-Shot]")
print(f"Response: {ask(few_shot_prompt)}")

# --- Compare both on multiple inputs ---
print("\n[Comparison on 3 inputs]")
test_texts = [
    "RAG systems are fascinating",
    "My virtual environment broke again",
    "The code runs without errors"
]

for text in test_texts:
    # Zero-shot
    zs = ask(f"Classify sentiment (Positive/Negative/Neutral): '{text}'")
    time.sleep(13)
    # Few-shot (reuse examples, change last text)
    fs_prompt = f"""Classify sentiment. Output ONLY: Positive, Negative, or Neutral.
Examples:
"I love this!" → Positive
"This is broken" → Negative
"It exists" → Neutral
Text: "{text}" →"""
    fs = ask(fs_prompt)
    time.sleep(13)
    print(f"\nText: '{text}'")
    print(f"  Zero-shot: {zs}")
    print(f"  Few-shot:  {fs}")

print("\n⏳ Waiting 15 seconds...")
time.sleep(15)

# ============================================================
# SECTION 2: Chain-of-Thought Prompting
# ============================================================

print("=" * 60)
print("SECTION 2: Chain-of-Thought Prompting")
print("=" * 60)

# Without chain-of-thought — just asks for answer
without_cot = ask("""A store sells AI books for $25 each.
They have a 20% discount today.
Abdelrhman buys 3 books.
How much does he pay?""")

print(f"\n[Without CoT]:\n{without_cot}")

# With chain-of-thought — forces step by step reasoning
with_cot = ask("""A store sells AI books for $25 each.
They have a 20% discount today.
Abdelrhman buys 3 books.
How much does he pay?

Think step by step:
1. Original price of 3 books
2. Calculate the discount amount
3. Calculate final price""")

print(f"\n[With CoT]:\n{with_cot}")

print("\n✅ Key insight: CoT reduces errors on multi-step problems")

print("\n⏳ Waiting 15 seconds...")
time.sleep(15)

# ============================================================
# SECTION 3: Role Prompting
# ============================================================

print("=" * 60)
print("SECTION 3: Role Prompting")
print("=" * 60)

question = "What is machine learning?"

# No role — generic answer
no_role = ask(question)
print(f"\n[No Role]:\n{no_role}\n")

# With role — completely different output
with_role = ask(
    question,
    system="You are a Cairo street vendor explaining tech to customers. Use simple Egyptian Arabic concepts and analogies. Maximum 3 sentences."
)
print(f"\n[Cairo Vendor Role]:\n{with_role}\n")

# Technical role
tech_role = ask(
    question,
    system="You are a senior ML engineer. Be precise and technical. Use proper terminology. Maximum 3 sentences."
)
print(f"\n[Senior ML Engineer Role]:\n{tech_role}")

print("\n✅ Key insight: Role prompting adapts output to specific audiences")

print("\n⏳ Waiting 15 seconds...")
time.sleep(15)

# ============================================================
# SECTION 4: Output Format Control
# ============================================================

print("=" * 60)
print("SECTION 4: Output Format Control")
print("=" * 60)

# Unstructured output — AI decides format
unstructured = ask("Tell me about RAG systems")
print(f"\n[Unstructured]:\n{unstructured}\n")

# JSON output — structured data
json_output = ask("""Analyze RAG systems and return ONLY a JSON object.
No explanation, no markdown, just valid JSON.
Format:
{
  "name": "technique name",
  "use_case": "what it's used for",
  "components": ["component1", "component2"],
  "difficulty": "beginner/intermediate/advanced"
}""")
print(f"\n[JSON Format]:\n{json_output}\n")

# How to use JSON output in real code:
import json
try:
    # Remove markdown code fences that AI often adds
    clean_json = json_output.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(clean_json)
    print(f"Parsed name: {parsed.get('name', 'N/A')}")
    print(f"Parsed difficulty: {parsed.get('difficulty', 'N/A')}")
except:
    print("JSON parsing skipped (output may have error)")

# Bullet points output
bullet_output = ask("""Explain RAG systems using exactly this format:
WHAT: [one sentence]
WHY: [one sentence]
HOW: [one sentence]
WHEN TO USE: [one sentence]""")
print(f"\n[Structured Format]:\n{bullet_output}")

print("\n✅ Key insight: Controlling output format enables programmatic use")

print("\n⏳ Waiting 15 seconds...")
time.sleep(15)

# ============================================================
# SECTION 5: Prompt Templates
# ============================================================

print("=" * 60)
print("SECTION 5: Prompt Templates")
print("=" * 60)

# A prompt template is a reusable prompt with variables
# Instead of writing new prompts every time — fill in the blanks

class PromptTemplate:
    """
    Reusable prompt with variable placeholders.
    Used in LangChain, LlamaIndex, and all major AI frameworks.
    """

    def __init__(self, template):
        """
        Args:
            template: String with {variable} placeholders
        """
        self.template = template

    def format(self, **kwargs):
        """
        Fills in the template variables.

        Args:
            **kwargs: Variable names and values

        Returns:
            Completed prompt string
        """
        return self.template.format(**kwargs)

    def run(self, **kwargs):
        """Formats template and sends to AI"""
        prompt = self.format(**kwargs)
        return ask(prompt)


# Template 1 — Sentiment Analysis
sentiment_template = PromptTemplate("""Classify the sentiment of this {content_type}.
Output ONLY one word: Positive, Negative, or Neutral.

{content_type}: "{text}"
Sentiment:""")

print("\n[Sentiment Template]")
print(sentiment_template.run(content_type="tweet", text="Just shipped my first RAG app!"))
print(sentiment_template.run(content_type="review", text="The documentation is confusing"))

# Template 2 — Concept Explainer
explainer_template = PromptTemplate("""Explain {concept} to a {audience}.
Use a {analogy_type} analogy.
Keep it under {max_words} words.""")

print("\n[Explainer Template]")
print(explainer_template.run(
    concept="vector embeddings",
    audience="10-year-old",
    analogy_type="food",
    max_words="50"
))

# Template 3 — Code Reviewer
review_template = PromptTemplate("""Review this {language} code for {focus}.
Be specific and actionable.
Code:
```{language}
{code}
```
Issues found:""")

print("\n[Code Review Template]")
sample_code = """
def divide(a, b):
    return a / b
"""
print(review_template.run(
    language="Python",
    focus="error handling",
    code=sample_code
))

print("\n✅ Key insight: Templates make prompts reusable and maintainable")


