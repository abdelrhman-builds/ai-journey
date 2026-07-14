# =============================================
# Day 45 - Object Detection with YOLO
# Author: Abdelrhman
# Date: July 2026
# =============================================
# Demonstrates real object detection (not just classification)
# using a pretrained YOLOv8 model — detecting multiple objects
# of different classes, with bounding boxes and confidence
# scores, in a single fast forward pass through the network.
# =============================================

from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO


# =============================================
# SECTION 1: Load Pretrained YOLO Model
# =============================================

def load_model():
    """
    Loads YOLOv8 (nano version - smallest, fastest variant)
    pretrained on the COCO dataset: 80 common object classes
    (person, car, dog, bus, etc.) learned from thousands of
    real-world labeled images.
    """
    model = YOLO('yolov8n.pt')
    print(f"Model loaded!")
    print(f"Classes this model can detect: {len(model.names)}")
    print(f"Sample classes: {list(model.names.values())[:15]}")
    return model


# =============================================
# SECTION 2: Run Detection on a Test Image
# =============================================

def detect_objects(model, image_url, conf_threshold=None):
    """
    Runs YOLO detection on an image from a URL.

    conf_threshold: if provided, only detections at or above
    this confidence score are returned. Production systems
    typically use 0.5 as a standard cutoff to filter out
    uncertain/borderline detections.
    """
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    if conf_threshold:
        results = model(image, conf=conf_threshold)
    else:
        results = model(image)

    print(f"\nDetected objects (conf_threshold={conf_threshold}):")
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2] bounding box
        print(f"  {class_name}: {round(confidence, 3)} confidence, "
              f"box: [{round(coords[0])}, {round(coords[1])}, "
              f"{round(coords[2])}, {round(coords[3])}]")

    return results


# =============================================
# SECTION 3: Visualize Results
# =============================================

def visualize_detection(results, filename='yolo_detection_result.png'):
    """
    Draws bounding boxes and labels directly onto the image
    and saves it — the standard way to visually confirm
    detection results.
    """
    result_image = results[0].plot()  # returns image with boxes drawn
    plt.figure(figsize=(10, 8))
    plt.imshow(result_image[:, :, ::-1])  # BGR to RGB conversion
    plt.axis('off')
    plt.title("YOLO Object Detection")
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    print(f"\nResult saved as {filename}")
    plt.close()


# =============================================
# SECTION 4: Run the Full Demo
# =============================================

if __name__ == "__main__":
    model = load_model()

    # Classic multi-object test image (bus + pedestrians)
    image_url = "https://ultralytics.com/images/bus.jpg"

    # Test 1: No confidence threshold — shows ALL detections,
    # including uncertain/borderline ones
    print("\n" + "=" * 60)
    print("TEST 1: No confidence threshold")
    print("=" * 60)
    results_all = detect_objects(model, image_url)
    visualize_detection(results_all, 'yolo_detection_all.png')

    # Test 2: With a realistic 0.5 confidence threshold —
    # shows only detections a production system would trust
    print("\n" + "=" * 60)
    print("TEST 2: Confidence threshold = 0.5 (production-realistic)")
    print("=" * 60)
    results_filtered = detect_objects(model, image_url, conf_threshold=0.5)
    visualize_detection(results_filtered, 'yolo_detection_filtered.png')