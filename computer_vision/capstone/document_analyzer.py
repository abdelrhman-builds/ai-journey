# =============================================
# Document & Scene Analyzer (v3 - Production Grade)
# Author: Abdelrhman
# Date: July 2026
# =============================================
# A robust, bilingual (Arabic + English) image analysis tool
# combining object detection, instance segmentation, and OCR.
# Designed for real-world use: proper error handling, confidence-
# aware classification, and a clean, importable API.
#
# Capabilities:
# - Detects and segments objects in scene images (80 COCO classes)
# - Extracts text from document images in English AND Arabic
# - Automatically classifies content as scene/document/mixed/unclear
# - Handles network failures, bad images, and empty results gracefully
# =============================================

from ultralytics import YOLO
import easyocr
import numpy as np
from PIL import Image
import requests
from io import BytesIO


# =============================================
# SECTION 1: Load All Models Once (Bilingual OCR)
# =============================================

print("Loading models (this happens once)...")

try:
    detector = YOLO('yolov8n.pt')
    segmenter = YOLO('yolov8n-seg.pt')
    # Single reader supporting BOTH English and Arabic —
    # EasyOCR can process multiple languages in one reader,
    # automatically handling mixed-language documents too
    ocr_reader = easyocr.Reader(['en', 'ar'])
    print("All models loaded successfully!\n")
except Exception as e:
    print(f"FATAL: Model loading failed: {e}")
    raise


# =============================================
# SECTION 2: Robust Image Loading
# =============================================

def load_image(image_source):
    """
    Loads an image from either a URL or a local file path,
    with proper error handling for common real-world failures:
    network errors, invalid URLs, corrupted files, unsupported
    formats.

    Returns None on failure instead of crashing — callers
    should check for this.
    """
    try:
        if image_source.startswith("http://") or image_source.startswith("https://"):
            response = requests.get(image_source, timeout=15)
            response.raise_for_status()  # raises an error for bad HTTP status codes
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_source)

        image = image.convert("RGB")
        return np.array(image)

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Could not download image from '{image_source}': {e}")
        return None
    except Exception as e:
        print(f"ERROR: Could not open image '{image_source}': {e}")
        return None


# =============================================
# SECTION 3: Object Analysis (Detection + Segmentation)
# =============================================

def run_object_analysis(image_np):
    """
    Runs detection + segmentation, returning structured
    object data. Wrapped in error handling since model
    inference can occasionally fail on unusual inputs.
    """
    try:
        detection_results = detector(image_np, conf=0.5, verbose=False)
        segmentation_results = segmenter(image_np, conf=0.5, verbose=False)

        objects_found = []
        for box in detection_results[0].boxes:
            class_id = int(box.cls[0])
            class_name = detector.names[class_id]
            confidence = float(box.conf[0])
            objects_found.append({"class": class_name, "confidence": round(confidence, 3)})

        return {
            "objects_detected": len(objects_found),
            "objects": objects_found,
            "detection_result": detection_results,
            "segmentation_result": segmentation_results,
            "error": None
        }
    except Exception as e:
        return {
            "objects_detected": 0,
            "objects": [],
            "detection_result": None,
            "segmentation_result": None,
            "error": str(e)
        }


# =============================================
# SECTION 4: Bilingual Text Analysis
# =============================================

def run_text_analysis(image_np):
    """
    Runs OCR supporting BOTH English and Arabic text in a
    single pass, returning structured text data with
    confidence scoring.

    KNOWN TRADEOFF (observed during testing): a bilingual
    reader gives more candidate character interpretations
    for ambiguous pixels, which occasionally introduces small
    errors on pure single-language text compared to a
    dedicated single-language reader (e.g. "World Health"
    read as "World Haalih" in one test, vs a perfect 0.998-
    confidence read from an English-only reader on the same
    phrase). This is an accepted tradeoff in exchange for
    handling both languages without needing separate readers.
    """
    try:
        results = ocr_reader.readtext(image_np)

        full_text = " ".join([text for (bbox, text, confidence) in results])
        avg_confidence = (
            sum(conf for (_, _, conf) in results) / len(results)
            if results else 0
        )

        return {
            "text_regions_found": len(results),
            "full_text": full_text,
            "word_count": len(full_text.split()),
            "average_confidence": round(avg_confidence, 3),
            "error": None
        }
    except Exception as e:
        return {
            "text_regions_found": 0,
            "full_text": "",
            "word_count": 0,
            "average_confidence": 0,
            "error": str(e)
        }


# =============================================
# SECTION 5: Confidence-Aware Classification
# =============================================

def classify_content(object_result, text_result,
                      text_word_threshold=10, text_confidence_threshold=0.5):
    """
    Classifies image content based on BOTH the QUANTITY and
    QUALITY of detected text, and the presence of detected
    objects. This avoids the common failure mode of treating
    low-confidence, garbled OCR fragments (e.g. from small
    angled signage in a photo) as genuine document text —
    verified directly: a bus image with garbled signage text
    (21 words, 0.267 confidence) was correctly classified as
    SCENE, not incorrectly flagged as containing "significant
    text," because it failed the confidence threshold.
    """
    has_objects = object_result["objects_detected"] > 0
    has_significant_text = (
        text_result["word_count"] >= text_word_threshold
        and text_result["average_confidence"] >= text_confidence_threshold
    )

    if has_objects and has_significant_text:
        return "mixed"
    elif has_objects:
        return "scene"
    elif has_significant_text:
        return "document"
    else:
        return "unclear"


# =============================================
# SECTION 6: The Unified, Robust Analysis Function
# =============================================

def analyze_image(image_source, verbose=True):
    """
    The main entry point: fetches an image (URL or local path),
    runs BOTH object analysis and bilingual text analysis
    unconditionally, classifies the content, and returns a
    complete structured result.

    Designed to NEVER crash on bad input — returns a result
    dict with an 'error' field instead, so calling code
    (e.g. a Streamlit app, an API endpoint, a batch script)
    can handle failures gracefully. Verified against a genuinely
    broken URL: returns {"success": False, "error": "..."}
    cleanly instead of raising an unhandled exception.
    """
    image_np = load_image(image_source)

    if image_np is None:
        return {
            "success": False,
            "error": f"Could not load image from: {image_source}",
            "content_type": None,
            "objects": None,
            "text": None
        }

    object_result = run_object_analysis(image_np)
    text_result = run_text_analysis(image_np)

    content_type = classify_content(object_result, text_result)

    result = {
        "success": True,
        "error": None,
        "content_type": content_type,
        "objects": object_result,
        "text": text_result
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Analyzing: {image_source}")
        print(f"{'='*60}")
        print(f"\nContent classification: {content_type.upper()}")

        print(f"\n--- Object Analysis ---")
        if object_result["error"]:
            print(f"Error: {object_result['error']}")
        else:
            print(f"Objects detected: {object_result['objects_detected']}")
            for obj in object_result['objects']:
                print(f"  {obj['class']}: {obj['confidence']} confidence")

        print(f"\n--- Text Analysis (English + Arabic) ---")
        if text_result["error"]:
            print(f"Error: {text_result['error']}")
        else:
            print(f"Text regions found: {text_result['text_regions_found']}")
            print(f"Word count: {text_result['word_count']}")
            print(f"Average OCR confidence: {text_result['average_confidence']}")
            if text_result['full_text']:
                preview = text_result['full_text'][:300]
                print(f"Extracted text (RAG-ready): {preview}"
                      f"{'...' if len(text_result['full_text']) > 300 else ''}")
            else:
                print("Extracted text: none found")

    return result


# =============================================
# SECTION 7: Demo — Scene, Document, and Error Handling
# =============================================

if __name__ == "__main__":
    # Test 1: Pure scene (with genuinely challenging garbled
    # low-confidence signage text — proves classification is
    # robust to this edge case)
    analyze_image("https://ultralytics.com/images/bus.jpg")

    # Test 2: Pure English document
    analyze_image(
        "https://raw.githubusercontent.com/JaidedAI/EasyOCR/master/examples/english.png"
    )

    # Test 3: A genuinely broken URL — proves error handling
    # works and the tool never crashes on bad input
    analyze_image("https://this-domain-does-not-exist-12345.com/fake.jpg")