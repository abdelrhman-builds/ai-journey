# =============================================
# Day 38 - Transformers Architecture (Attention)
# Author: Abdelrhman
# Date: July 2026
# =============================================
# Visualizes real attention weights from a pretrained
# BERT model, and demonstrates encoder vs decoder
# architectural differences.
# =============================================

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel


# =============================================
# SECTION 1: Load BERT and Extract Real Attention
# =============================================

def visualize_bert_attention(sentence, layer=5, head=0):
    """
    Loads a real pretrained BERT model and extracts its
    ACTUAL learned attention weights for a given sentence.

    layer: which of BERT's 12 transformer layers to inspect
           (different layers learn different types of patterns —
           early layers often focus on local/syntactic patterns,
           later layers often focus on semantic/coreference patterns)
    head: which of the 12 attention heads within that layer
          (different heads specialize in different relationship types)
    """
    print(f"\n{'='*60}")
    print(f"BERT ATTENTION VISUALIZATION")
    print(f"Sentence: '{sentence}'")
    print(f"Layer: {layer}, Head: {head}")
    print(f"{'='*60}")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # output_attentions=True is CRITICAL — without this flag,
    # BERT computes attention internally but doesn't return it
    model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
    model.eval()  # inference mode, not training

    inputs = tokenizer(sentence, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    with torch.no_grad():  # don't compute gradients, we're not training
        outputs = model(**inputs)

    # outputs.attentions is a tuple of 12 tensors (one per layer)
    # each tensor shape: (batch, num_heads, seq_len, seq_len)
    attention = outputs.attentions[layer][0, head].numpy()

    print(f"\nTokens: {tokens}")
    print(f"Attention matrix shape: {attention.shape}")
    print(f"(This is a {len(tokens)}x{len(tokens)} matrix — every")
    print(f"token's attention score toward every other token)")

    # Show which tokens each token attends to MOST
    print(f"\nTop attention connections:")
    for i, token in enumerate(tokens):
        if token in ['[CLS]', '[SEP]']:
            continue  # skip special tokens for clarity
        top_attended_idx = attention[i].argmax()
        top_score = attention[i][top_attended_idx]
        print(f"  '{token}' attends most to '{tokens[top_attended_idx]}' (score: {round(float(top_score), 3)})")

    # Create heatmap visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention, xticklabels=tokens, yticklabels=tokens,
                cmap='viridis', annot=False, cbar_kws={'label': 'Attention Weight'})
    plt.title(f"BERT Attention Heatmap (Layer {layer}, Head {head})\n'{sentence}'")
    plt.xlabel("Attending TO (Key)")
    plt.ylabel("Attending FROM (Query)")
    plt.tight_layout()
    plt.savefig('bert_attention_heatmap.png', dpi=100)
    print(f"\n✅ Heatmap saved as bert_attention_heatmap.png")
    plt.close()

    return attention, tokens


# =============================================
# SECTION 2: The Coreference Test (The "it" Example)
# =============================================

def coreference_test():
    """
    Tests whether BERT's attention actually resolves the
    classic "it" ambiguity problem discussed in Concept 2.
    """
    sentence = "The animal didn't cross the street because it was too tired"

    print(f"\n{'='*60}")
    print("COREFERENCE TEST — Does attention resolve 'it'?")
    print(f"{'='*60}")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
    model.eval()

    inputs = tokenizer(sentence, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    with torch.no_grad():
        outputs = model(**inputs)

    # Find the index of "it" in our tokenized sentence
    it_idx = tokens.index('it')

    print(f"\nTokens: {tokens}")
    print(f"'it' is at position {it_idx}")

    # Check attention from "it" across ALL layers to find
    # which layer/head resolves the reference most clearly
    print(f"\nWhat does 'it' attend to most, across different layers?")
    for layer in [0, 3, 6, 9, 11]:
        # Average across all 12 heads in this layer
        layer_attention = outputs.attentions[layer][0].mean(dim=0)
        it_attention = layer_attention[it_idx].numpy()
        top_idx = it_attention.argmax()
        print(f"  Layer {layer}: 'it' attends most to '{tokens[top_idx]}' "
              f"(score: {round(float(it_attention[top_idx]), 3)})")




def coreference_test_per_head():
    """
    Instead of averaging across all 12 heads (which can dilute
    a signal present in only 1-2 specialized heads), check EVERY
    individual head at a specific layer to find which ones, if any,
    show 'it' attending to 'animal'.
    """
    sentence = "The animal didn't cross the street because it was too tired"

    print(f"\n{'='*60}")
    print("REFINED TEST — Checking Individual Heads (Not Averaged)")
    print(f"{'='*60}")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
    model.eval()

    inputs = tokenizer(sentence, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    with torch.no_grad():
        outputs = model(**inputs)

    it_idx = tokens.index('it')
    animal_idx = tokens.index('animal')
    street_idx = tokens.index('street')

    print(f"\nChecking layer 8 (commonly cited for coreference), all 12 heads:")
    layer_8_attention = outputs.attentions[8][0]  # shape: (12 heads, seq, seq)

    for head in range(12):
        it_to_animal = float(layer_8_attention[head][it_idx][animal_idx])
        it_to_street = float(layer_8_attention[head][it_idx][street_idx])
        print(f"  Head {head}: 'it'->'animal' = {round(it_to_animal, 3)}, "
              f"'it'->'street' = {round(it_to_street, 3)}")
        

# =============================================
# SECTION 3: Encoder vs Decoder Demonstration
# =============================================

def encoder_vs_decoder_demo():
    """
    Demonstrates the practical difference between encoder
    (BERT, bidirectional) and decoder (GPT-2, causal/unidirectional)
    architectures using their actual attention patterns.
    """
    print(f"\n{'='*60}")
    print("ENCODER vs DECODER ARCHITECTURE")
    print(f"{'='*60}")

    sentence = "AI is transforming"

    # BERT (Encoder) - bidirectional
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
    bert_model.eval()

    bert_inputs = bert_tokenizer(sentence, return_tensors='pt')
    bert_tokens = bert_tokenizer.convert_ids_to_tokens(bert_inputs['input_ids'][0])

    with torch.no_grad():
        bert_outputs = bert_model(**bert_inputs)

    # Check if first token can "see" later tokens (bidirectional test)
    first_word_idx = 1  # after [CLS]
    last_layer_attention = bert_outputs.attentions[-1][0].mean(dim=0)
    attention_from_first = last_layer_attention[first_word_idx].numpy()

    print(f"\nBERT (Encoder) - '{sentence}':")
    print(f"Tokens: {bert_tokens}")
    print(f"Attention FROM first real word '{bert_tokens[first_word_idx]}' TO all tokens:")
    for i, tok in enumerate(bert_tokens):
        print(f"  -> '{tok}': {round(float(attention_from_first[i]), 3)}")
    print("Notice: attention to LATER words is NOT zero —")
    print("BERT can see the FULL sentence in both directions.")

    # GPT-2 (Decoder) - causal/unidirectional
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2', output_attentions=True)
    gpt2_model.eval()

    gpt2_inputs = gpt2_tokenizer(sentence, return_tensors='pt')
    gpt2_tokens = gpt2_tokenizer.convert_ids_to_tokens(gpt2_inputs['input_ids'][0])

    with torch.no_grad():
        gpt2_outputs = gpt2_model(**gpt2_inputs, output_attentions=True)

    first_layer_attention = gpt2_outputs.attentions[-1][0].mean(dim=0)
    attention_from_first_gpt = first_layer_attention[0].numpy()

    print(f"\nGPT-2 (Decoder) - '{sentence}':")
    print(f"Tokens: {gpt2_tokens}")
    print(f"Attention FROM first word '{gpt2_tokens[0]}' TO all tokens:")
    for i, tok in enumerate(gpt2_tokens):
        print(f"  -> '{tok}': {round(float(attention_from_first_gpt[i]), 3)}")
    print("Notice: attention to LATER words IS zero (or near-zero) —")
    print("GPT-2 can ONLY see PREVIOUS words (causal masking).")
    print("This is why GPT generates text left-to-right, one token")
    print("at a time, and can never 'peek' at what it hasn't generated yet.")


# =============================================
# SECTION 4: Run All Demos
# =============================================

if __name__ == "__main__":
    visualize_bert_attention(
        "The cat sat on the mat because it was comfortable",
        layer=5, head=0
    )
    coreference_test()
    encoder_vs_decoder_demo()
    coreference_test_per_head()