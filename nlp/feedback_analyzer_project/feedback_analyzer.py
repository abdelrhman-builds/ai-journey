# =============================================
# Bilingual Customer Feedback Analyzer
# Author: Abdelrhman
# Date: July 2026
# =============================================
# A capstone project combining Week 6's NLP work:
# - Language detection (Arabic vs English)
# - Language-appropriate sentiment analysis (Day 40's
#   finding: AraBERT for Arabic, standard model for English)
# - Named entity extraction
# - Zero-shot feedback categorization
# Produces a structured analysis report for each piece
# of customer feedback — directly usable for a Khamsat
# "feedback analysis" service offering.
# =============================================

import re
from transformers import pipeline


# =============================================
# SECTION 1: Language Detection
# =============================================

def detect_language(text):
    """
    Simple, fast language detection based on Unicode ranges.
    Arabic characters fall in the U+0600-U+06FF block.
    This avoids needing a separate heavy language-detection
    model for a binary Arabic/English decision.
    """
    arabic_pattern = re.compile(r'[\u0600-\u06FF]')
    arabic_chars = len(arabic_pattern.findall(text))
    total_chars = len(text.replace(" ", ""))

    if total_chars == 0:
        return "unknown"

    arabic_ratio = arabic_chars / total_chars
    return "arabic" if arabic_ratio > 0.3 else "english"


# =============================================
# SECTION 2: Load All Models Once
# =============================================
# Following the established pattern (Day 23 FastAPI startup,
# Day 33 agent tools): expensive model loading happens ONCE,
# not per-analysis-call.

print("Loading models (this happens once)...")

sentiment_en = pipeline("sentiment-analysis")
sentiment_ar = pipeline(
    "sentiment-analysis",
    model="CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"
)
ner_en = pipeline("ner", aggregation_strategy="simple")
ner_ar = pipeline(
    "ner",
    model="CAMeL-Lab/bert-base-arabic-camelbert-mix-ner",
    aggregation_strategy="simple"
)
zero_shot = pipeline("zero-shot-classification")
# Note: zero-shot classification model used here is English-
# trained; for a production Arabic zero-shot system, an
# Arabic-specific model would be a natural next improvement —
# flagged honestly rather than glossed over.

print("✅ All models loaded!\n")


# =============================================
# SECTION 3: The Core Analysis Function
# =============================================

def analyze_feedback(text):
    """
    Runs a complete analysis pipeline on one piece of
    customer feedback, routing to the correct language-specific
    models based on Day 40's proven finding that specialized
    models significantly outperform generic multilingual ones.
    """
    language = detect_language(text)

    # Route to the correct sentiment model
    if language == "arabic":
        sentiment_result = sentiment_ar(text)[0]
        entities = ner_ar(text)
    else:
        sentiment_result = sentiment_en(text)[0]
        entities = ner_en(text)

    # Zero-shot categorization (language-agnostic labels,
    # works reasonably on both since labels are in English
    # and the model has some multilingual capability)
    categories = ["complaint", "praise", "question", "suggestion", "bug report"]
    category_result = zero_shot(text, categories)
    top_category = category_result['labels'][0]
    top_category_score = category_result['scores'][0]

    return {
        "text": text,
        "language": language,
        "sentiment_label": sentiment_result['label'],
        "sentiment_confidence": round(sentiment_result['score'], 3),
        "entities": [(e['word'], e['entity_group']) for e in entities],
        "category": top_category,
        "category_confidence": round(top_category_score, 3),
    }


# =============================================
# SECTION 4: Batch Report Generator
# =============================================

def generate_report(feedback_list):
    """
    Analyzes a batch of feedback and prints a structured report —
    the kind of output a real client would receive from this
    as a delivered service.
    """
    print("=" * 70)
    print("CUSTOMER FEEDBACK ANALYSIS REPORT")
    print("=" * 70)

    results = [analyze_feedback(text) for text in feedback_list]

    for i, r in enumerate(results, 1):
        print(f"\n--- Feedback #{i} ---")
        print(f"Text: {r['text']}")
        print(f"Language detected: {r['language']}")
        print(f"Sentiment: {r['sentiment_label']} (confidence: {r['sentiment_confidence']})")
        print(f"Category: {r['category']} (confidence: {r['category_confidence']})")
        if r['entities']:
            print(f"Entities mentioned: {r['entities']}")
        else:
            print("Entities mentioned: none detected")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    languages = [r['language'] for r in results]
    print(f"Total feedback analyzed: {len(results)}")
    print(f"Arabic: {languages.count('arabic')} | English: {languages.count('english')}")

    categories_count = {}
    for r in results:
        categories_count[r['category']] = categories_count.get(r['category'], 0) + 1
    print(f"Category breakdown: {categories_count}")

    return results


# =============================================
# SECTION 5: Demo Run
# =============================================

if __name__ == "__main__":
    sample_feedback = [
        "The RAG system works perfectly and Google Gemini integration is smooth!",
        "الخدمة ممتازة والفريق محترف جداً، أنصح بالتعامل معهم",
        "My PDF upload keeps failing, this is frustrating",
        "هل يمكن إضافة دعم لملفات Excel في التحديث القادم؟",
        # "Can you add Excel file support in the next update?"
        "Can you add support for CSV files? It would be really helpful",
    ]

    generate_report(sample_feedback)