# =============================================
# Day 36 - Tokenization + Text Preprocessing
# Author: Abdelrhman
# Date: July 2026
# =============================================
# Explores how different tokenizers split text into
# tokens — the foundation of ALL modern NLP systems.
# =============================================

import nltk
from transformers import (
    AutoTokenizer,        # loads any HuggingFace tokenizer
    BertTokenizer,        # BERT's WordPiece tokenizer
    GPT2Tokenizer         # GPT-2's BPE tokenizer
)

# Download NLTK data (only needed once)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)


# =============================================
# SECTION 1: Word Tokenization (Classic NLP)
# =============================================

def word_tokenize_demo(text):
    """
    NLTK's word tokenizer — splits on spaces and punctuation.
    This is the simplest, oldest approach. Used before deep
    learning became dominant in NLP (~2018).
    """
    tokens = nltk.word_tokenize(text)
    print(f"\n--- Word Tokenization (NLTK) ---")
    print(f"Input:  {text}")
    print(f"Tokens: {tokens}")
    print(f"Count:  {len(tokens)} tokens")
    return tokens


# =============================================
# SECTION 2: BERT Tokenization (WordPiece)
# =============================================

def bert_tokenize_demo(text):
    """
    BERT uses WordPiece tokenization — splits unknown/rare
    words into subword pieces. Pieces after the first start
    with ## to show they're continuations.

    Example: "tokenization" → ["token", "##ization"]
    This lets BERT handle words it never saw in training
    by breaking them into familiar subparts.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.encode(text)

    print(f"\n--- BERT Tokenization (WordPiece) ---")
    print(f"Input:  {text}")
    print(f"Tokens: {tokens}")
    print(f"IDs:    {ids}")
    print(f"Count:  {len(tokens)} tokens")
    return tokens, ids


# =============================================
# SECTION 3: GPT-2 Tokenization (BPE)
# =============================================

def gpt2_tokenize_demo(text):
    """
    GPT-2 uses Byte Pair Encoding (BPE) — starts with
    individual characters, then merges the most frequent
    pairs repeatedly until reaching the target vocabulary size.

    The Ġ prefix (Ġ = space) indicates tokens that start
    a new word. This is how BPE encodes word boundaries
    without a separate space token.
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.encode(text)

    print(f"\n--- GPT-2 Tokenization (BPE) ---")
    print(f"Input:  {text}")
    print(f"Tokens: {tokens}")
    print(f"IDs:    {ids}")
    print(f"Count:  {len(tokens)} tokens")
    return tokens, ids


# =============================================
# SECTION 4: Text Preprocessing Pipeline
# =============================================

def preprocess_text(text):
    """
    Classic NLP preprocessing pipeline — used before
    feeding text into traditional ML models (not needed
    for modern transformers which handle raw text).
    Still useful for search, filtering, and analysis tasks.
    """
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer

    print(f"\n--- Text Preprocessing Pipeline ---")
    print(f"Original: {text}")

    # Step 1: Lowercase
    text_lower = text.lower()
    print(f"Lowercase: {text_lower}")

    # Step 2: Tokenize
    tokens = nltk.word_tokenize(text_lower)
    print(f"Tokenized: {tokens}")

    # Step 3: Remove punctuation and numbers
    tokens_clean = [t for t in tokens if t.isalpha()]
    print(f"Clean tokens: {tokens_clean}")

    # Step 4: Remove stopwords (common words with little meaning)
    # Stopwords: "the", "is", "at", "which", "on", etc.
    stop_words = set(stopwords.words('english'))
    tokens_no_stop = [t for t in tokens_clean if t not in stop_words]
    print(f"Without stopwords: {tokens_no_stop}")

    # Step 5: Stemming (reduce words to their root form)
    # "running" -> "run", "studies" -> "studi"
    # Note: stemming is aggressive and may produce non-words
    stemmer = PorterStemmer()
    tokens_stemmed = [stemmer.stem(t) for t in tokens_no_stop]
    print(f"Stemmed: {tokens_stemmed}")

    return tokens_stemmed


# =============================================
# SECTION 5: Compare Tokenizers Side by Side
# =============================================

def compare_tokenizers(text):
    """
    Shows how the SAME text is tokenized differently
    by different tokenizers — critical for understanding
    why model choice affects token count, cost, and behavior.
    """
    print(f"\n{'='*60}")
    print(f"COMPARING TOKENIZERS ON: '{text}'")
    print(f"{'='*60}")

    # NLTK word tokenizer
    nltk_tokens = nltk.word_tokenize(text)

    # BERT WordPiece
    bert_tok = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_tokens = bert_tok.tokenize(text)

    # GPT-2 BPE
    gpt2_tok = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_tokens = gpt2_tok.tokenize(text)

    print(f"\nNLTK (word):     {nltk_tokens}")
    print(f"Tokens: {len(nltk_tokens)}")

    print(f"\nBERT (WordPiece): {bert_tokens}")
    print(f"Tokens: {len(bert_tokens)}")

    print(f"\nGPT-2 (BPE):     {gpt2_tokens}")
    print(f"Tokens: {len(gpt2_tokens)}")

    print(f"\nKey insight: same text = different token counts")
    print(f"This affects: model input length, API cost,")
    print(f"context window usage, and generation speed.")


# =============================================
# SECTION 6: Run All Demos
# =============================================

if __name__ == "__main__":
    sample_text = "AI Engineering is transforming how we build software in Cairo and beyond."
    tricky_text = "Tokenization is the foundational preprocessing step in NLP pipelines."

    # Basic demos
    word_tokenize_demo(sample_text)
    bert_tokenize_demo(tricky_text)
    gpt2_tokenize_demo(tricky_text)

    # Preprocessing pipeline
    preprocess_text(sample_text)

    # Side-by-side comparison
    compare_tokenizers(tricky_text)