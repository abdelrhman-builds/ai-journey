# 🧪 Prompt Engineering Experiments
### Day 10 — AI Engineering Journey | Date: 9 June 2026

---

## Overview

This file is my prompt engineering lab notebook.
Every experiment documents: hypothesis, prompt used, result, and conclusion.
This is how professional AI engineers build reliable prompts.

---

## Experiment 1 — Zero-Shot vs Few-Shot

### Hypothesis
Few-shot produces more consistent, predictable output format than zero-shot.

### Prompts Tested

**Zero-shot prompt:**
```
Classify the sentiment of this text.
Text: "I just deployed my first AI application!"
Sentiment:
```

**Few-shot prompt:**
```
Classify sentiment. Output ONLY one word: Positive, Negative, or Neutral.

Examples:
Text: "I love building AI apps!" → Positive
Text: "This bug is so frustrating" → Negative
Text: "The model returned a response" → Neutral

Text: "I just deployed my first AI application!"
Sentiment:
```

### Results

| Input | Zero-Shot Output | Few-Shot Output |
|-------|-----------------|-----------------|
| "I just deployed my first AI app!" | `Sentiment: **Positive**` | `Positive` |
| "RAG systems are fascinating" | `Classify sentiment...\n**Positive**` | `Positive` |
| "My virtual environment broke again" | Long explanation with bullet points | `Negative` |
| "The code runs without errors" | `Positive` | `Positive` |

### Analysis

```
Zero-shot problems:
→ Inconsistent format — sometimes includes "Sentiment:" prefix
→ Sometimes adds bold formatting (**Positive**)
→ Sometimes writes a full paragraph explanation
→ Format changes unpredictably between calls

Few-shot advantages:
→ Always outputs exactly one word
→ Copies the format from examples exactly
→ Predictable — same format every time
→ Safe to use programmatically
```

### Conclusion
✅ Few-shot wins decisively for format consistency.

### Rules to Remember
```
Use zero-shot when: format doesn't matter, quick exploration
Use few-shot when:  need exact output format, classification tasks,
                    output will be parsed programmatically
```

---

## Experiment 2 — Chain-of-Thought Prompting

### Hypothesis
Chain-of-thought (CoT) reduces errors and improves structure on multi-step problems.

### Prompts Tested

**Without CoT:**
```
A store sells AI books for $25 each.
They have a 20% discount today.
Abdelrhman buys 3 books.
How much does he pay?
```

**With CoT:**
```
A store sells AI books for $25 each.
They have a 20% discount today.
Abdelrhman buys 3 books.
How much does he pay?

Think step by step:
1. Original price of 3 books
2. Calculate the discount amount
3. Calculate final price
```

### Results

**Without CoT output:**
```
1. Calculate the discount per book: 20% of $25 = $5
2. Price per book after discount: $25 - $5 = $20
3. Total for 3 books: 3 × $20 = $60
Answer: Abdelrhman pays $60.
```

**With CoT output:**
```
1. Original price of 3 books: $25 × 3 = $75
2. Discount amount: 20% of $75 = $15
3. Final price: $75 - $15 = $60
Abdelrhman pays $60.
```

### Analysis

```
Both approaches got the correct answer: $60 ✅

Difference:
Without CoT → AI still showed steps (gemini-2.5-flash is powerful)
With CoT    → Steps are clearer, numbered, easier to verify

Important insight:
CoT matters MORE with:
→ Weaker/smaller models (they need explicit guidance)
→ Complex multi-step problems (long reasoning chains)
→ Math with many operations
→ Logic puzzles

With powerful models like gemini-2.5-flash:
→ CoT still improves structure and verifiability
→ Even when answer would be correct without it
```

### Conclusion
✅ CoT improves structure and reduces errors, especially on complex problems.

### Rules to Remember
```
Always use CoT for:   math, logic, multi-step analysis, debugging
Add to any prompt:    "Think step by step" or numbered steps
Verify each step:     you can catch errors before the final answer
```

---

## Experiment 3 — Role Prompting

### Hypothesis
Defining a role dramatically changes the tone, complexity, and style of output.

### Prompts Tested

**No role system prompt:**
```
What is machine learning?
```

**Cairo vendor role:**
```
System: You are a Cairo street vendor explaining tech to customers.
        Use simple Egyptian Arabic concepts and analogies.
        Maximum 3 sentences.
User: What is machine learning?
```

**Senior ML engineer role:**
```
System: You are a senior ML engineer. Be precise and technical.
        Use proper terminology. Maximum 3 sentences.
User: What is machine learning?
```

### Results

**No role output:**
```
500+ word explanation with headers, bullet points,
comparisons, examples — completely generic
```

**Cairo vendor output:**
```
يا معلم، ده زي ما تكون بتعلّم القرد يلقط الموز لوحده
من غير ما كل مرة تقول له "خد دي". الكمبيوتر بنعلمه
يشوف الحاجات ويتعلم منها زي ما انت بتتعلم من خبرتك
في السوق. بيفهم لوحده إيه اللي ماشي وإيه اللي مش ماشي.
```
[Responded in ARABIC with Egyptian street market analogy!]

**Senior ML engineer output:**
```
Machine learning is a subfield of AI that focuses on developing
algorithms allowing systems to learn from data without explicit
programming. These algorithms identify patterns and make
data-driven predictions, improving performance through experience.
It encompasses supervised, unsupervised, and reinforcement learning.
```

### Analysis

```
No role:       Generic, verbose, one-size-fits-all
Cairo vendor:  Egyptian Arabic, street market analogy, perfect for
               non-technical Egyptian audience
Senior ML:     Technical, precise, terminology-heavy, 3 sentences

The Cairo vendor response is remarkable:
→ Used "يا معلم" (common Egyptian greeting)
→ Used monkey/banana analogy (Egyptian humor style)
→ Responded entirely in Arabic
→ This is YOUR competitive advantage as an Egyptian developer
```

### Conclusion
✅ Role prompting is the most powerful tool for audience-specific output.

### Competitive Advantage
```
As an Egyptian AI engineer:
→ You can build Arabic-language AI applications
→ Role prompting makes the AI respond naturally in Arabic
→ Egyptian market understands Egyptian analogies
→ This is a moat that foreign competitors can't easily replicate
```

### Rules to Remember
```
Always define a role when:
→ Writing for a specific audience (beginner vs expert)
→ Need a specific language or dialect
→ Want a specific personality or tone
→ Building customer-facing applications

Role format:
"You are [persona]. [behavior rules]. [constraints]."
```

---

## Experiment 4 — Output Format Control

### Hypothesis
Explicit format instructions produce structured, parseable output.

### Prompts Tested

**Unstructured:**
```
Tell me about RAG systems in 3 sentences.
```

**JSON format:**
```
Analyze RAG systems and return ONLY a JSON object.
No explanation, no markdown, just valid JSON.
Format: {"name": ..., "use_case": ..., "components": [...], "difficulty": ...}
```

**Structured WHAT/WHY/HOW/WHEN:**
```
Explain RAG systems using exactly this format:
WHAT: [one sentence]
WHY: [one sentence]
HOW: [one sentence]
WHEN TO USE: [one sentence]
```

### Results

**Unstructured output:**
```
RAG systems enhance LLMs by retrieving relevant information from
an external knowledge base before generating a response. This
grounds the model's output in factual data, reducing hallucinations.
By combining retrieval with generation, RAG provides more accurate answers.
```

**JSON output:**
```json
{
  "name": "Retrieval-Augmented Generation (RAG)",
  "use_case": "Enhancing factual accuracy and reducing hallucinations in LLMs",
  "components": [
    "Retriever",
    "Generator",
    "Knowledge Base/Corpus",
    "Embeddings",
    "Vector Database"
  ],
  "difficulty": "intermediate"
}
```

**Structured output:**
```
WHAT: RAG combines retrieval and generation to provide accurate responses.
WHY: They reduce hallucinations by grounding responses in real documents.
HOW: A retriever fetches relevant chunks, then LLM generates from context.
WHEN TO USE: When you need accurate, verifiable answers from specific documents.
```

### Critical Production Issue — JSON Markdown Wrapping

```python
# Problem: AI wraps JSON in markdown code fences
# Raw response:
# ```json
# {"name": "RAG", ...}
# ```

# This BREAKS json.loads():
import json
json.loads(response.text)  # ❌ SyntaxError — can't parse markdown

# Fix — always clean before parsing:
raw = response.text
clean = raw.replace("```json", "").replace("```", "").strip()
data = json.loads(clean)   # ✅ works perfectly
print(data["name"])        # "Retrieval-Augmented Generation (RAG)"
print(data["difficulty"])  # "intermediate"
```

### Conclusion
✅ Format control works reliably for all three format types.

### Rules to Remember
```
For human reading:    unstructured is fine
For programmatic use: always specify exact format
For JSON output:      always clean markdown fences before json.loads()
For APIs/webhooks:    structured format prevents parsing errors

Production tip:
Use response_format={"type": "json_object"} when available
→ Forces pure JSON without markdown fences
```

---

## Experiment 5 — Prompt Templates

### Hypothesis
Reusable templates reduce repetition and ensure consistency across multiple calls.

### PromptTemplate Class Built

```python
class PromptTemplate:
    def __init__(self, template):
        self.template = template     # string with {variable} placeholders

    def format(self, **kwargs):
        return self.template.format(**kwargs)   # fill in variables

    def run(self, **kwargs):
        prompt = self.format(**kwargs)
        return ask(prompt)                      # format + send to AI
```

### Templates Built and Results

**Template 1 — Sentiment Analysis:**
```python
sentiment_template = PromptTemplate("""Classify the sentiment of this {content_type}.
Output ONLY one word: Positive, Negative, or Neutral.
{content_type}: "{text}"
Sentiment:""")

sentiment_template.run(content_type="tweet",
                       text="Just shipped my first RAG app!")
# → "Positive" ✅

sentiment_template.run(content_type="review",
                       text="The documentation is confusing")
# → "Negative" ✅
```

**Template 2 — Concept Explainer:**
```python
explainer_template = PromptTemplate("""Explain {concept} to a {audience}.
Use a {analogy_type} analogy.
Keep it under {max_words} words.""")

explainer_template.run(
    concept="vector embeddings",
    audience="10-year-old",
    analogy_type="food",
    max_words="50"
)
# → "Imagine all your favorite snacks. A vector embedding is like
#    a secret code for each snack, telling us if it's crunchy,
#    sweet, or salty. It helps us find similar snacks super fast!" ✅
```

**Template 3 — Code Reviewer:**
```python
review_template = PromptTemplate("""Review this {language} code for {focus}.
Code:
```{language}
{code}
```
Issues found:""")

review_template.run(language="Python", focus="error handling", code=divide_func)
# → Detailed ZeroDivisionError and TypeError analysis ✅
# → Provided 3 fix options with code examples
```

### Conclusion
✅ Templates dramatically reduce code repetition and ensure consistency.

### Rules to Remember
```
Build templates for:
→ Any prompt used more than twice
→ Prompts with variable parts (names, topics, languages)
→ Production prompts that need consistent format

Template naming:
sentiment_template   → describes the task
explainer_template   → describes the task
review_template      → describes the task

This is exactly how LangChain's PromptTemplate works internally.
```

---

## Master Rules — Day 10 Summary

| Pattern | Use When | Avoid When | Example Use Case |
|---------|----------|------------|-----------------|
| Zero-shot | Quick exploration, flexible format | Need exact output format | "Summarize this text" |
| Few-shot | Classification, exact format needed | Prompt gets too long | Sentiment analysis |
| Chain-of-thought | Math, logic, multi-step | Simple factual questions | Word problems, debugging |
| Role prompting | Audience-specific, tone control | Generic responses fine | Arabic customer support |
| Format control | Programmatic use, APIs | Human reading only | JSON extraction |
| Templates | Repeated patterns, production | One-off prompts | Production classifiers |

---

## Prompt Engineering Golden Rules

```
1. Be specific — vague prompts produce vague output
   Bad:  "Tell me about Python"
   Good: "Explain Python's list comprehension in 2 sentences for a beginner"

2. Show examples — few-shot beats zero-shot for format consistency
   Always include 2-3 examples for classification tasks

3. Define the role — persona changes everything
   "You are a..." at the start of every system prompt

4. Specify the format — don't let AI decide
   "Output ONLY valid JSON" / "Respond in exactly this format:"

5. Use templates — never write the same prompt twice
   Build a PromptTemplate for any prompt used more than twice

6. Test with edge cases — find where your prompt breaks
   Normal input → works
   Empty input  → test it
   Weird input  → test it
   Long input   → test it

7. Document everything — this file IS your competitive advantage
   What worked, what failed, why — build institutional knowledge
```

---

## Technical Notes

### Rate Limits on Free Tier (gemini-2.5-flash)
```
Per minute: 5 requests
Per day:    20 requests

Strategy:
→ time.sleep(15) between calls = 4 calls/minute (safe)
→ Plan experiments: 20 requests = enough for one full session
→ Create new API key when daily limit hit
→ Never waste requests on repeated identical calls
```

### JSON Parsing Pattern (Use in Every Project)
```python
import json

def parse_json_response(response_text):
    """Safely parse JSON from AI response"""
    try:
        # Remove markdown code fences if present
        clean = response_text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse JSON: {e}"}
```

---

*Day 10 of 90 — Prompt engineering foundation complete. 🚀*
