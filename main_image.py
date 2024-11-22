import os
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import matplotlib.pyplot as plt

# Initialize OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Load YOLO models
vehicle_model = YOLO('yolo11m.pt')
model_path = os.path.join('.', 'runs', 'detect', 'train12', 'weights', 'best.pt')
license_plate_model = YOLO(model_path)

# Load the image
image_path = 'test5.jpg'  # Update this to your image path
frame = cv2.imread(image_path)

if frame is None:
    print("Error: Could not load image.")
    exit()

# Detect vehicles
vehicle_detections = vehicle_model(frame)[0]

# Define class IDs for vehicles (adjust as needed)
vehicles = [2, 3, 5, 7]  # Example class IDs for cars, buses, trucks, etc.
for vehicle in vehicle_detections.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = vehicle
    if int(class_id) in vehicles:
        # Draw bounding box around vehicle
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Detect license plates in the vehicle area
        vehicle_frame = frame[int(y1):int(y2), int(x1):int(x2)]
        license_plate_detections = license_plate_model(vehicle_frame)[0]
        
        for plate in license_plate_detections.boxes.data.tolist():
            px1, py1, px2, py2, pscore, pclass_id = plate
            
            # Adjust coordinates for the plate relative to the vehicle's bounding box
            lx1, ly1, lx2, ly2 = int(x1 + px1), int(y1 + py1), int(x1 + px2), int(y1 + py2)
            
            # Draw bounding box around license plate
            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 0, 255), 2)

            # Crop and read the license plate using OCR
            plate_crop = frame[ly1:ly2, lx1:lx2]
            plate_text = ""
            
            if plate_crop.size > 0:
                plate_text = reader.readtext(plate_crop, detail=0)
            
            if plate_text:
                # Display the text above the vehicle and in the zoomed overlay
                cv2.putText(frame, plate_text[0], (lx1, ly1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Create an enlarged overlay of the license plate and text
                zoomed_plate = cv2.resize(plate_crop, (300, 100))
                frame[0:100, 0:300] = zoomed_plate  # Top-left corner for the zoomed overlay
                cv2.putText(frame, plate_text[0], (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

# Display the image using matplotlib
# Convert BGR to RGB for correct color display in matplotlib
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

plt.imshow(frame_rgb)
plt.title("License Plate Detection")
plt.axis("off")  # Turn off axis
plt.show()

# Optional: Save the output image
output_image_path = 'output_image_detected.jpg'
cv2.imwrite(output_image_path, frame)
print(f"Image saved to {output_image_path}")
