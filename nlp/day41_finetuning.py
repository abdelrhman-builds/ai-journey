# =============================================
# Day 41 - Fine-tuning a Model on Custom Data
# Author: Abdelrhman
# Date: July 2026
# =============================================
# Minimal fine-tuning example: adapts a small pretrained
# model to classify short support messages into 3 custom
# categories not present in any standard dataset.
# =============================================

from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
import numpy as np


# =============================================
# SECTION 1: Tiny Custom Dataset
# =============================================
# In a REAL fine-tuning project, you'd need hundreds/thousands
# of labeled examples. This tiny set (9 examples) is enough
# to demonstrate the MECHANISM working, not to achieve strong
# real-world accuracy — that distinction is important.

def build_dataset():
    """
    3 custom categories that don't exist in any standard
    pretrained model: specific to a hypothetical Khamsat
    client's exact support categories.
    """
    texts = [
        "My file upload keeps failing after 50%", "upload_issue",
        "The PDF won't process, stuck loading forever", "upload_issue",
        "Can't get my document to upload to the system", "upload_issue",

        "How much does the premium plan cost?", "pricing_question",
        "What's included in the $20 package?", "pricing_question",
        "Is there a discount for students?", "pricing_question",

        "The AI gave me a wrong answer about my contract", "accuracy_issue",
        "The system said information that isn't in my document", "accuracy_issue",
        "Answers don't match what's actually in the file", "accuracy_issue",
    ]

    pairs = [(texts[i], texts[i+1]) for i in range(0, len(texts), 2)]
    all_texts = [p[0] for p in pairs]
    all_labels_str = [p[1] for p in pairs]

    label_map = {"upload_issue": 0, "pricing_question": 1, "accuracy_issue": 2}
    all_labels = [label_map[l] for l in all_labels_str]

    return all_texts, all_labels, label_map


# =============================================
# SECTION 2: Fine-tuning Setup
# =============================================

def finetune_demo():
    print("=" * 60)
    print("FINE-TUNING DEMONSTRATION")
    print("=" * 60)

    texts, labels, label_map = build_dataset()
    reverse_label_map = {v: k for k, v in label_map.items()}

    print(f"\nDataset: {len(texts)} examples, {len(label_map)} categories")
    print(f"Categories: {list(label_map.keys())}")

    # Use a SMALL pretrained model — faster to fine-tune,
    # good enough for this demonstration
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # num_labels=3 tells the model to add a NEW classification
    # head with 3 output categories — this new head starts
    # RANDOM and gets trained; the rest of the model (language
    # understanding) starts from its PRETRAINED weights
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label_map)
    )

    # Tokenize all texts (Day 36 concepts, applied directly)
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length",
                         truncation=True, max_length=32)

    dataset = Dataset.from_dict({"text": texts, "label": labels})
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # =============================================
    # SECTION 3: Training Configuration
    # =============================================

    training_args = TrainingArguments(
        output_dir="./finetuned_model",
        num_train_epochs=10,        # small dataset needs more passes
        per_device_train_batch_size=4,
        logging_steps=1,
        save_strategy="no",         # don't save checkpoints (demo only)
        report_to="none"            # disable wandb/other logging integrations
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("\nTraining (fine-tuning) starting...")
    trainer.train()
    print("\n✅ Fine-tuning complete!")

    # =============================================
    # SECTION 4: Test the Fine-tuned Model
    # =============================================

    print("\n--- Testing on NEW messages (not in training data) ---")

    test_messages = [
        "My upload is stuck and won't finish",
        "Do you have a cheaper plan available?",
        "The bot told me something that wasn't in my file"
    ]

    model.eval()
    import torch

    for msg in test_messages:
        inputs = tokenizer(msg, return_tensors="pt", truncation=True, max_length=32)
        with torch.no_grad():
            outputs = model(**inputs)
        predicted_class = outputs.logits.argmax(-1).item()
        confidence = torch.softmax(outputs.logits, dim=-1).max().item()

        print(f"\nMessage: '{msg}'")
        print(f"Predicted category: {reverse_label_map[predicted_class]} "
              f"(confidence: {round(confidence, 3)})")


if __name__ == "__main__":
    finetune_demo()