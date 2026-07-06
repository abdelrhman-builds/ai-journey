# =============================================
# Day 37 - TF-IDF + Word Embeddings (Word2Vec)
# Author: Abdelrhman
# Date: July 2026
# =============================================
# Builds the conceptual bridge between simple word
# counting and modern transformer embeddings.
# TF-IDF: statistical importance | Word2Vec: neural meaning
# =============================================

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import nltk

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


# =============================================
# SECTION 1: TF-IDF
# =============================================

def tfidf_demo():
    """
    TF-IDF assigns importance scores to words based on
    how often they appear in ONE document vs ALL documents.

    High TF-IDF score: word appears often in this doc
                       but rarely in other docs = important
    Low TF-IDF score:  word appears in most docs = common,
                       not specific to this doc
    """
    # Our "corpus" — a small collection of documents
    corpus = [
        "AI Engineering is building production AI systems in Cairo",
        "RAG systems retrieve documents and generate accurate answers",
        "Cairo is the capital city of Egypt with many engineers",
        "LangChain helps build AI agents and RAG pipelines in Python",
        "Egypt has a growing tech scene with AI and software engineers"
    ]

    print("=" * 60)
    print("TF-IDF DEMONSTRATION")
    print("=" * 60)
    print("\nCorpus:")
    for i, doc in enumerate(corpus):
        print(f"  Doc {i}: {doc}")

    # Create TF-IDF matrix
    # max_features=20: only keep top 20 most important words
    # stop_words='english': automatically remove common words
    vectorizer = TfidfVectorizer(max_features=20, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Get feature names (the words in our vocabulary)
    feature_names = vectorizer.get_feature_names_out()
    print(f"\nVocabulary ({len(feature_names)} words): {list(feature_names)}")

    # Show TF-IDF scores for each document
    print("\nTF-IDF Scores per Document:")
    for i, doc in enumerate(corpus):
        scores = tfidf_matrix[i].toarray()[0]
        # Get top 5 words by TF-IDF score for this document
        top_indices = scores.argsort()[-5:][::-1]
        top_words = [(feature_names[j], round(scores[j], 3))
                     for j in top_indices if scores[j] > 0]
        print(f"\n  Doc {i}: '{doc[:40]}...'")
        print(f"  Top words: {top_words}")

    # Show a specific word's importance across documents
    print("\n\nWord 'AI' score across all documents:")
    if 'ai' in feature_names:
        ai_idx = list(feature_names).index('ai')
        for i in range(len(corpus)):
            score = tfidf_matrix[i, ai_idx]
            print(f"  Doc {i}: {round(score, 3)}")

    # Document similarity using TF-IDF vectors
    print("\n\nDocument Similarity (cosine similarity):")
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)
    print("Query: 'Doc 0' (AI Engineering in Cairo)")
    for i in range(1, len(corpus)):
        sim = round(similarity_matrix[0][i], 3)
        print(f"  Similarity with Doc {i}: {sim} — '{corpus[i][:40]}...'")

    return vectorizer, tfidf_matrix


# =============================================
# SECTION 2: Word2Vec
# =============================================

def word2vec_demo():
    """
    Word2Vec learns dense vector representations of words
    by training a neural network to predict context words.

    Two training modes:
    CBOW (Continuous Bag of Words): predict the center word
         from surrounding context words
    Skip-gram: predict surrounding context words from the
         center word (better for rare words)

    After training, words used in similar contexts have
    similar vectors — this encodes SEMANTIC MEANING.
    """
    print("\n" + "=" * 60)
    print("WORD2VEC DEMONSTRATION")
    print("=" * 60)

    # Training corpus — we need enough text for Word2Vec
    # to learn meaningful relationships. In practice,
    # Word2Vec is trained on billions of words.
    sentences = [
        # AI and tech sentences
        ["AI", "engineering", "builds", "production", "systems"],
        ["machine", "learning", "trains", "models", "on", "data"],
        ["deep", "learning", "uses", "neural", "networks"],
        ["python", "is", "the", "language", "of", "AI"],
        ["LangChain", "builds", "AI", "agents", "with", "tools"],
        ["RAG", "retrieves", "documents", "to", "answer", "questions"],
        ["transformers", "are", "the", "foundation", "of", "modern", "AI"],

        # Egyptian cities — so the model learns geographic relationships
        ["Cairo", "is", "the", "capital", "city", "of", "Egypt"],
        ["Alexandria", "is", "a", "coastal", "city", "in", "Egypt"],
        ["Egypt", "is", "a", "country", "in", "North", "Africa"],
        ["Cairo", "has", "millions", "of", "people", "and", "engineers"],
        ["Alexandria", "has", "a", "famous", "ancient", "library"],

        # King/Queen analogy — classic Word2Vec example
        ["the", "king", "rules", "the", "kingdom"],
        ["the", "queen", "rules", "the", "kingdom"],
        ["the", "king", "is", "a", "man", "with", "power"],
        ["the", "queen", "is", "a", "woman", "with", "power"],
        ["the", "man", "went", "to", "work", "in", "the", "city"],
        ["the", "woman", "went", "to", "work", "in", "the", "city"],
        ["prince", "is", "a", "young", "man", "royal"],
        ["princess", "is", "a", "young", "woman", "royal"],
    ]

    # Train Word2Vec model
    model = Word2Vec(
        sentences=sentences,
        vector_size=50,    # dimension of each word vector
                           # (real models use 300-1536+ dims)
        window=3,          # context window: look 3 words
                           # left and right of each word
        min_count=1,       # include words seen at least once
                           # (real models use min_count=5+)
        workers=1,         # single thread for reproducibility
        epochs=200,        # train 200 passes over the data
                           # (more epochs = better learning
                           #  for small datasets like this)
        seed=42            # for reproducible results
    )

    print(f"\nModel trained!")
    print(f"Vocabulary size: {len(model.wv)} words")
    print(f"Vector size: {model.vector_size} dimensions")

    # =============================================
    # Demo 1: Word Similarity
    # =============================================
    print("\n--- Word Similarity ---")
    print("(Higher score = more similar meaning)")

    pairs = [
        ("Cairo", "Alexandria"),    # both Egyptian cities
        ("Cairo", "Egypt"),         # city and its country
        ("king", "queen"),          # both royalty
        ("king", "man"),            # king is a male role
        ("AI", "machine"),          # related tech concepts
        ("AI", "Cairo"),            # unrelated (should be low)
    ]

    for word1, word2 in pairs:
        try:
            similarity = model.wv.similarity(word1, word2)
            print(f"  {word1} <-> {word2}: {round(similarity, 3)}")
        except KeyError as e:
            print(f"  {e} not in vocabulary")

    # =============================================
    # Demo 2: Most Similar Words
    # =============================================
    print("\n--- Most Similar Words ---")

    test_words = ["Cairo", "king", "AI", "woman"]
    for word in test_words:
        try:
            similar = model.wv.most_similar(word, topn=3)
            similar_str = [(w, round(s, 3)) for w, s in similar]
            print(f"  Words similar to '{word}': {similar_str}")
        except KeyError:
            print(f"  '{word}' not in vocabulary")

    # =============================================
    # Demo 3: The Famous Analogy
    # =============================================
    print("\n--- Vector Arithmetic (The Famous Analogy) ---")
    print("Testing: king - man + woman = ?")
    print("(Small dataset - result approximate)")

    try:
        # positive=['king', 'woman'] means add these vectors
        # negative=['man'] means subtract this vector
        result = model.wv.most_similar(
            positive=['king', 'woman'],
            negative=['man'],
            topn=3
        )
        print(f"king - man + woman = {[(w, round(s, 3)) for w, s in result]}")
        print("\nNote: With only 20 training sentences, the analogy")
        print("may not work perfectly. Real Word2Vec is trained on")
        print("billions of words — the relationship becomes clear")
        print("at scale. The PRINCIPLE is what matters here.")
    except KeyError as e:
        print(f"KeyError: {e}")

    # =============================================
    # Demo 4: Vector Distance Between Cairo and Alexandria
    # =============================================
    print("\n--- Geographic Relationship in Vector Space ---")
    try:
        cairo_vec = model.wv['Cairo']
        alex_vec = model.wv['Alexandria']
        egypt_vec = model.wv['Egypt']

        # Cosine similarity
        from numpy.linalg import norm
        def cosine_sim(v1, v2):
            return np.dot(v1, v2) / (norm(v1) * norm(v2))

        print(f"Cairo <-> Alexandria: {round(cosine_sim(cairo_vec, alex_vec), 3)}")
        print(f"Cairo <-> Egypt:      {round(cosine_sim(cairo_vec, egypt_vec), 3)}")
        print(f"Cairo <-> AI:         {round(cosine_sim(cairo_vec, model.wv['AI']), 3)}")
        print("\nCities should be more similar to each other")
        print("than to unrelated concepts — even in our tiny model.")
    except KeyError as e:
        print(f"KeyError: {e}")

    return model


# =============================================
# SECTION 3: Connect to Week 3 RAG Embeddings
# =============================================

def connect_to_rag():
    """
    Explains the conceptual connection between Word2Vec
    and the embeddings we used in Week 3's RAG system.
    """
    print("\n" + "=" * 60)
    print("CONNECTION TO WEEK 3 RAG EMBEDDINGS")
    print("=" * 60)
    print("""
Word2Vec (today):
- Trained on our 20 sentences
- 50-dimensional word vectors
- Each WORD gets one vector
- Similar words -> similar vectors

sentence-transformers/all-MiniLM-L6-v2 (Week 3 RAG):
- Trained on billions of sentences
- 384-dimensional sentence vectors
- Each SENTENCE/CHUNK gets one vector
- Similar meanings -> similar vectors

The core idea is IDENTICAL:
similar meaning = similar position in vector space

The differences are scale and sophistication:
Word2Vec: word-level, count-based context
Modern transformers: sentence-level, attention-based context

When we called vectorstore.similarity_search(question, k=3)
in Week 3, we were doing EXACTLY what Word2Vec similarity
does — finding the vectors closest to our query vector.
The math is the same. The training is vastly more powerful.
""")


# =============================================
# SECTION 4: Run All Demos
# =============================================

if __name__ == "__main__":
    tfidf_demo()
    model = word2vec_demo()
    connect_to_rag()