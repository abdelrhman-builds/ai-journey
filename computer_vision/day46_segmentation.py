# =============================================
# Day 46 - Instance Segmentation with YOLO
# Author: Abdelrhman
# Date: July 2026
# =============================================
# Compares pixel-level instance segmentation against Day 45's
# bounding-box object detection on the SAME test image,
# concretely proving how much precision segmentation adds.
# =============================================

from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO


# =============================================
# SECTION 1: Load Segmentation Model
# =============================================

def load_segmentation_model():
    """
    Loads YOLOv8's segmentation variant ('-seg' suffix).
    Unlike the standard detection model (Day 45), this outputs
    a pixel-level MASK for each detected object, not just a
    rectangular bounding box.
    """
    model = YOLO('yolov8n-seg.pt')
    print(f"Segmentation model loaded!")
    print(f"Classes: {len(model.names)}")
    return model


# =============================================
# SECTION 2: Run Segmentation
# =============================================

def segment_objects(model, image_url, conf_threshold=0.5):
    """
    Runs instance segmentation on an image, printing each
    detected object's class, confidence, and the actual pixel
    count of its mask (not just a bounding box).
    """
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    results = model(image, conf=conf_threshold)

    print(f"\nSegmented objects (conf_threshold={conf_threshold}):")
    detections = []
    for i, box in enumerate(results[0].boxes):
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        # Get the pixel mask for this specific detected object
        mask = results[0].masks.data[i]
        mask_pixel_count = int(mask.sum().item())

        print(f"  {class_name}: {round(confidence, 3)} confidence, "
              f"mask covers {mask_pixel_count} pixels")

        detections.append({
            "class": class_name,
            "confidence": confidence,
            "mask_pixels": mask_pixel_count
        })

    return results, detections


# =============================================
# SECTION 3: Compare Bounding Box vs Mask Precision
# =============================================

def compare_box_vs_mask(box_coords, mask_pixel_count, object_name="object"):
    """
    Concretely proves how much "extra" background area a
    bounding box includes compared to the object's true shape,
    by comparing box AREA (width x height) against the actual
    mask pixel count.

    box_coords: [x1, y1, x2, y2] from a detection result
    mask_pixel_count: actual pixel count from a segmentation result
    """
    box_width = box_coords[2] - box_coords[0]
    box_height = box_coords[3] - box_coords[1]
    box_area = box_width * box_height

    coverage_pct = 100 * mask_pixel_count / box_area
    background_pct = 100 - coverage_pct

    print(f"\n{object_name} bounding box area: {box_area} pixels")
    print(f"{object_name} actual mask area:  {mask_pixel_count} pixels")
    print(f"Mask covers {round(coverage_pct, 1)}% of the bounding box")
    print(f"The remaining {round(background_pct, 1)}% of the box "
          f"is BACKGROUND, not actually part of the {object_name}")

    return coverage_pct


# =============================================
# SECTION 4: Visualize Segmentation Results
# =============================================

def visualize_segmentation(results, filename='yolo_segmentation_result.png'):
    """
    Draws pixel masks (and bounding boxes for reference)
    directly onto the image and saves it.
    """
    result_image = results[0].plot()
    plt.figure(figsize=(10, 8))
    plt.imshow(result_image[:, :, ::-1])  # BGR to RGB conversion
    plt.axis('off')
    plt.title("YOLO Instance Segmentation")
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    print(f"\nResult saved as {filename}")
    plt.close()


# =============================================
# SECTION 5: Run the Full Demo
# =============================================

if __name__ == "__main__":
    model_seg = load_segmentation_model()

    # Same test image used in Day 45 for a direct, fair comparison
    image_url = "https://ultralytics.com/images/bus.jpg"

    results, detections = segment_objects(model_seg, image_url, conf_threshold=0.5)
    visualize_segmentation(results)

    # Direct comparison: Day 45's bus bounding box vs today's actual mask
    bus_box_from_day45 = [23, 231, 805, 757]
    bus_mask_pixels = next(d["mask_pixels"] for d in detections if d["class"] == "bus")

    compare_box_vs_mask(bus_box_from_day45, bus_mask_pixels, object_name="Bus")