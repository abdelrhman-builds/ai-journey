# =============================================
# Day 39 - Using Pretrained Models (HuggingFace Pipeline)
# Author: Abdelrhman
# Date: July 2026
# =============================================
# Applies everything learned Days 36-38 (tokenization,
# embeddings, attention) to real, practical NLP tasks
# using HuggingFace's pipeline() interface.
# =============================================

from transformers import pipeline


# =============================================
# SECTION 1: Sentiment Analysis
# =============================================

def sentiment_analysis_demo():
    """
    Demonstrates sentiment analysis with TWO models:
    1. English-only default (fails on Arabic)
    2. Multilingual model (correctly handles Arabic)

    This comparison is intentional — it proves why model
    selection matters for non-English use cases, directly
    relevant to Arabic market positioning.
    """
    print("=" * 60)
    print("SENTIMENT ANALYSIS")
    print("=" * 60)

    # English-only model (HuggingFace's default)
    classifier_en = pipeline("sentiment-analysis")

    texts = [
        "This RAG system works perfectly and saves me so much time!",
        "The API documentation was confusing and hard to follow.",
        "I'm not sure if this product is good or bad, it's okay I guess.",
    ]

    print("\n--- English-only model ---")
    for text in texts:
        result = classifier_en(text)[0]
        print(f"\nText: '{text}'")
        print(f"Sentiment: {result['label']} (confidence: {round(result['score'], 3)})")

    # Multilingual model (handles Arabic correctly)
    classifier_multi = pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )

    arabic_text = "أنا سعيد جداً بهذا المنتج"

    print("\n--- Comparison: Arabic text ---")
    result_en = classifier_en(arabic_text)[0]
    result_multi = classifier_multi(arabic_text)[0]

    print(f"\nText: '{arabic_text}' (means: 'I am very happy with this product')")
    print(f"English-only model result:  {result_en['label']} ({round(result_en['score'], 3)}) — WRONG")
    print(f"Multilingual model result:  {result_multi['label']} ({round(result_multi['score'], 3)}) — CORRECT")


# =============================================
# SECTION 2: Named Entity Recognition (NER)
# =============================================

def ner_demo():
    """
    Identifies and classifies named entities in text:
    PER (person), ORG (organization), LOC (location), MISC.
    """
    print("\n" + "=" * 60)
    print("NAMED ENTITY RECOGNITION (NER)")
    print("=" * 60)

    # aggregation_strategy="simple" replaces the older
    # grouped_entities=True parameter in newer transformers versions.
    # It merges sub-word tokens (e.g. "Cai", "##ro") back into
    # complete entity names (e.g. "Cairo").
    ner = pipeline("ner", aggregation_strategy="simple")

    text = ("Abdelrhman works on AI projects using Google Gemini "
             "and LangChain. He is based in Cairo, Egypt, and has "
             "built systems deployed on HuggingFace Spaces.")

    entities = ner(text)

    print(f"\nText: '{text}'")
    print(f"\nEntities found:")
    for entity in entities:
        print(f"  {entity['word']} -> {entity['entity_group']} "
              f"(confidence: {round(entity['score'], 3)})")


# =============================================
# SECTION 3: Zero-Shot Classification
# =============================================

def zero_shot_demo():
    """
    Classifies text into categories WITHOUT any training
    examples for those specific categories — the model uses
    its general language understanding to match text to
    label descriptions.

    Real business use case: categorizing support tickets,
    routing customer inquiries, content moderation — without
    needing to collect and label thousands of training examples
    for every possible category.
    """
    print("\n" + "=" * 60)
    print("ZERO-SHOT CLASSIFICATION")
    print("=" * 60)

    classifier = pipeline("zero-shot-classification")

    text = "I need help uploading my PDF file to your system, it keeps failing"

    candidate_labels = ["technical support", "billing question",
                        "feature request", "complaint", "compliment"]

    result = classifier(text, candidate_labels)

    print(f"\nText: '{text}'")
    print(f"\nCandidate labels: {candidate_labels}")
    print(f"\nRanked results:")
    for label, score in zip(result['labels'], result['scores']):
        print(f"  {label}: {round(score, 3)}")


# =============================================
# SECTION 4: Text Summarization
# =============================================

def summarization_demo():
    """
    Condenses long text into a shorter summary.

    NOTE: This Colab environment's transformers version has
    a restructured pipeline task registry that doesn't include
    "summarization" as a recognized task name. We bypass this
    by loading the model and tokenizer directly instead of
    going through pipeline() — which is literally what
    pipeline() does internally anyway.
    """
    print("\n" + "=" * 60)
    print("TEXT SUMMARIZATION")
    print("=" * 60)

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    long_text = """
    Retrieval-Augmented Generation (RAG) is a technique that combines
    information retrieval with text generation to produce more accurate
    and grounded AI responses. Instead of relying solely on what a
    language model learned during training, RAG systems first search
    through a knowledge base of documents to find relevant information,
    then use that retrieved information as context for generating an
    answer. This approach significantly reduces hallucination, where
    AI models confidently state incorrect information, because the
    model is instructed to only use the retrieved context rather than
    its own potentially outdated or incorrect internal knowledge.
    RAG systems typically work by first splitting documents into smaller
    chunks, converting those chunks into numerical vector representations
    called embeddings, and storing them in a specialized vector database.
    When a user asks a question, the system converts the question into
    the same type of embedding, searches the vector database for the
    most similar chunks, and passes those chunks along with the original
    question to a language model to generate a final, grounded answer.
    """

    # Tokenize the input (Day 36 concepts, applied directly)
    inputs = tokenizer(long_text, max_length=1024, truncation=True, return_tensors="pt")

    # Generate the summary (encoder-decoder: encode input, decode summary)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=60,
        min_length=20,
        num_beams=4,          # beam search: considers 4 possible
                               # sequences at once, picks the best
        early_stopping=True
    )

    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print(f"\nOriginal text ({len(long_text.split())} words):")
    print(long_text.strip())
    print(f"\nSummary ({len(summary_text.split())} words):")
    print(summary_text)


# =============================================
# SECTION 5: Run All Demos
# =============================================

if __name__ == "__main__":
    sentiment_analysis_demo()
    ner_demo()
    zero_shot_demo()
    summarization_demo()