# =============================================
# Day 47 - OCR with EasyOCR
# Author: Abdelrhman
# Date: July 2026
# =============================================
# Demonstrates Optical Character Recognition (OCR) for both
# English and Arabic text, directly addressing the scanned-
# document gap identified in Week 3's RAG system (Day 18) —
# OCR converts image-based text into machine-readable text
# ready for ingestion into the existing RAG pipeline.
# =============================================

import easyocr
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import numpy as np


# =============================================
# SECTION 1: English OCR
# =============================================

def load_english_reader():
    """
    Initializes EasyOCR for English. Downloads detection and
    recognition models on first use (deep learning models,
    built on CNN + sequence architectures).
    """
    reader = easyocr.Reader(['en'])
    print("English OCR reader loaded!")
    return reader


def run_ocr_on_url(reader, image_url):
    """
    Runs OCR on an image fetched from a URL. Returns a list
    of (bounding_box, text, confidence) tuples — one per
    detected text region.
    """
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    image_np = np.array(image)

    results = reader.readtext(image_np)

    print(f"Detected {len(results)} text regions:\n")
    for (bbox, text, confidence) in results:
        print(f"  Text: '{text}' (confidence: {round(confidence, 3)})")

    return results


def reconstruct_document_text(ocr_results):
    """
    Combines all detected text regions into one coherent
    document string — exactly the format needed to feed
    into the existing RAG pipeline's load_text() tool
    (Day 33), turning a scanned image into searchable content.
    """
    full_text = " ".join([text for (bbox, text, confidence) in ocr_results])
    print("\nReconstructed document text (ready for RAG ingestion):")
    print(full_text)
    print(f"\nWord count: {len(full_text.split())}")
    return full_text


# =============================================
# SECTION 2: Arabic OCR
# =============================================

def load_arabic_reader():
    """Initializes EasyOCR for Arabic script."""
    reader = easyocr.Reader(['ar'])
    print("Arabic OCR reader loaded!")
    return reader


def create_arabic_test_image(text, filename='arabic_test.png',
                               font_path="/usr/share/fonts/truetype/kacst/KacstOne.ttf"):
    """
    Generates a test image containing Arabic text, using a
    font that correctly renders Arabic script shaping
    (connected letters, position-dependent letter forms).
    """
    img = Image.new('RGB', (600, 200), color='white')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, 30)
    draw.text((20, 80), text, fill='black', font=font)
    img.save(filename)
    return np.array(img)


# =============================================
# SECTION 3: Run the Full Demo
# =============================================

if __name__ == "__main__":
    # English OCR test
    reader_en = load_english_reader()

    image_url = "https://raw.githubusercontent.com/JaidedAI/EasyOCR/master/examples/english.png"
    results_en = run_ocr_on_url(reader_en, image_url)
    reconstruct_document_text(results_en)

    # Arabic OCR test
    reader_ar = load_arabic_reader()

    arabic_text = "الذكاء الاصطناعي يغير العالم"  # "AI is changing the world"
    image_ar_np = create_arabic_test_image(arabic_text)

    results_ar = reader_ar.readtext(image_ar_np)
    print(f"\nDetected {len(results_ar)} Arabic text regions:\n")
    for (bbox, text, confidence) in results_ar:
        print(f"  Text: '{text}' (confidence: {round(confidence, 3)})")