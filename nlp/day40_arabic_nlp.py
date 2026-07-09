# =============================================
# Day 40 - Arabic NLP + AraBERT
# Author: Abdelrhman
# Date: July 2026
# =============================================
# Compares generic multilingual models against AraBERT
# (Arabic-specialized BERT) on real Arabic NLP tasks,
# demonstrating the competitive differentiator this
# journey has been building toward since Day 14.
# =============================================

from transformers import pipeline, AutoTokenizer, AutoModel
import pyarabic.araby as araby


# =============================================
# SECTION 1: Arabic Text Preprocessing
# =============================================

def arabic_preprocessing_demo():
    """
    Demonstrates Arabic-specific text normalization —
    steps that don't exist in English NLP pipelines at all.
    """
    print("=" * 60)
    print("ARABIC TEXT PREPROCESSING")
    print("=" * 60)

    # Text with diacritics (تشكيل) — short vowel marks
    text_with_diacritics = "كَتَبَ الطَّالِبُ الدَّرْسَ"
    text_without = "كتب الطالب الدرس"

    print(f"\nWith diacritics:    {text_with_diacritics}")
    print(f"Without diacritics: {text_without}")

    # Strip diacritics using pyarabic — most real-world
    # text and most models expect diacritic-free text
    stripped = araby.strip_diacritics(text_with_diacritics)
    print(f"\nStripped result:    {stripped}")
    print(f"Matches expected:   {stripped == text_without}")

    # Normalize different letter forms that represent the
    # same sound but are written differently — a common
    # source of "missed matches" in Arabic search/NLP
    variations = ["إسلام", "اسلام", "أسلام"]
    print(f"\nLetter variations of the same word:")
    for v in variations:
        normalized = araby.normalize_hamza(v)
        print(f"  '{v}' -> normalized: '{normalized}'")


# =============================================
# SECTION 2: Sentiment Analysis - Multilingual vs AraBERT
# =============================================

def sentiment_comparison():
    """
    Direct comparison on REALISTIC Egyptian business text —
    not just simple sentences, but the kind of text a real
    Khamsat client's customer reviews might contain.
    """
    print("\n" + "=" * 60)
    print("SENTIMENT: MULTILINGUAL vs ARABERT")
    print("=" * 60)

    test_reviews = [
        "الخدمة ممتازة والفريق محترف جداً، أنصح بالتعامل معهم",
        # "The service is excellent and the team is very professional, I recommend dealing with them"
        "للأسف المنتج مش زي ما كان متوقع، جودة ضعيفة",
        # "Unfortunately the product isn't as expected, poor quality"
        "التطبيق بطيء جداً ومليء بالأخطاء البرمجية",
        # "The app is very slow and full of bugs"
    ]

    # Generic multilingual model (Day 39's approach)
    multi_classifier = pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )

    # AraBERT fine-tuned for sentiment (Arabic-specialized)
    arabert_classifier = pipeline(
        "sentiment-analysis",
        model="CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"
    )

    for review in test_reviews:
        print(f"\nReview: '{review}'")

        multi_result = multi_classifier(review)[0]
        print(f"  Multilingual model: {multi_result['label']} ({round(multi_result['score'], 3)})")

        arabert_result = arabert_classifier(review)[0]
        print(f"  AraBERT (specialized): {arabert_result['label']} ({round(arabert_result['score'], 3)})")


# =============================================
# SECTION 3: Named Entity Recognition on Arabic Text
# =============================================

def arabic_ner_demo():
    """
    Tests entity extraction on a realistic Arabic business
    sentence — the kind of content that would appear in a
    contract, invoice, or company document.
    """
    print("\n" + "=" * 60)
    print("ARABIC NAMED ENTITY RECOGNITION")
    print("=" * 60)

    text = "شركة أمازون افتتحت مقرها الجديد في مدينة القاهرة بمصر"
    # "Amazon company opened its new headquarters in Cairo city, Egypt"

    ner = pipeline(
        "ner",
        model="CAMeL-Lab/bert-base-arabic-camelbert-mix-ner",
        aggregation_strategy="simple"
    )

    entities = ner(text)

    print(f"\nText: '{text}'")
    print(f"(Translation: 'Amazon company opened its new headquarters in Cairo city, Egypt')")
    print(f"\nEntities found:")
    for entity in entities:
        print(f"  {entity['word']} -> {entity['entity_group']} "
              f"(confidence: {round(entity['score'], 3)})")


# =============================================
# SECTION 4: Run All Demos
# =============================================

if __name__ == "__main__":
    arabic_preprocessing_demo()
    sentiment_comparison()
    arabic_ner_demo()