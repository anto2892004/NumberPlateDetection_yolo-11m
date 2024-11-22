import os
import cv2
import numpy as np
from ultralytics import YOLO
from util import get_car, read_license_plate, write_csv
from sort.sort import Sort
import matplotlib.pyplot as plt
import pytesseract  # Assuming you are using pytesseract for OCR

# Initialize tracker and results dictionary
results = {}
mot_tracker = Sort()

# Load YOLO models
coco_model = YOLO('yolo11m.pt')
model_path = os.path.join('.', 'runs', 'detect', 'train12', 'weights', 'best.pt')
license_plate_detector = YOLO(model_path)

# Define vehicle classes to detect
vehicles = [2, 3, 5, 7]  # Example class IDs for cars, buses, trucks, etc.

# Load the image
image_path = 'test5.jpg'  # Update this to your image path
frame = cv2.imread(image_path)

if frame is None:
    print("Error: Could not load image.")
    exit()

# Initialize a dictionary to store results for the image
frame_nmr = 0  # Single frame/image
results[frame_nmr] = {}

# Detect vehicles with confidence threshold adjustment
detections = coco_model(frame, conf=0.3)[0]  # Adjust confidence if needed
detections_ = []

# Print and filter vehicle detections
for detection in detections.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = detection
    print(f"Class ID: {class_id}, Score: {score}, BBox: ({x1}, {y1}, {x2}, {y2})")
    if int(class_id) in vehicles:
        detections_.append([x1, y1, x2, y2, score])

# Visualize vehicle detections
for x1, y1, x2, y2, _ in detections_:
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

# Track vehicles
track_ids = mot_tracker.update(np.asarray(detections_))
print("Tracked vehicle IDs:", [track[4] for track in track_ids])

# Detect license plates
license_plates = license_plate_detector(frame)[0]
for i, license_plate in enumerate(license_plates.boxes.data.tolist()):
    x1, y1, x2, y2, score, class_id = license_plate

    # Draw license plate bounding box for visualization
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Assign license plate to car
    xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
    print("Assigned Car ID:", car_id)

    # Crop and preprocess license plate image
    license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

    # Resize image to improve OCR accuracy
    license_plate_crop_resized = cv2.resize(license_plate_crop_gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    
    # Adaptive thresholding to enhance OCR results
    license_plate_crop_thresh = cv2.adaptiveThreshold(
        license_plate_crop_resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8
    )

    # Optional dilation to improve character separation
    kernel = np.ones((1, 1), np.uint8)
    license_plate_crop_thresh = cv2.dilate(license_plate_crop_thresh, kernel, iterations=1)

    # Debugging: Save the preprocessed license plate for manual inspection
    debug_plate_path = f"./debug_license_plate_{i}.jpg"
    cv2.imwrite(debug_plate_path, license_plate_crop_thresh)
    print(f"Saved preprocessed license plate to {debug_plate_path}")

    # OCR Configuration
    ocr_config = "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    
    # Read license plate number using pytesseract
    license_plate_text = pytesseract.image_to_string(license_plate_crop_thresh, config=ocr_config).strip()
    print("Detected License Plate Text:", license_plate_text)

    if license_plate_text:
        results[frame_nmr][car_id] = {
            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
            'license_plate': {
                'bbox': [x1, y1, x2, y2],
                'text': license_plate_text,
                'bbox_score': score,
                'text_score': None  # Assuming score is not available from pytesseract
            }
        }

# Option 1: Save the image and view it manually
output_image_path = './detected_image.jpg'
cv2.imwrite(output_image_path, frame)
print(f"Image saved to {output_image_path}. You can open this file to view the detections.")

# Option 2: Display the image inline with matplotlib (useful for Jupyter notebooks)
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
plt.imshow(frame_rgb)
plt.axis('off')  # Hide axes
plt.show()

# Write results to a CSV file
output_csv_path = './output_results.csv'
if any(results[frame_nmr]):  # Write only if there are results
    write_csv(results, output_csv_path)
    print(f"Results written to {output_csv_path}")
else:
    print("No results to write to CSV.")
