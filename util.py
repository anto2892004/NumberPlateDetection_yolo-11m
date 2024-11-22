import os
import cv2
import string
import numpy as np
import pytesseract
import easyocr

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)  # GPU can be set to True if available

# Character conversion dictionaries
dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                car_data = results[frame_nmr][car_id]
                if 'car' in car_data.keys() and 'license_plate' in car_data.keys():
                    license_number = car_data['license_plate'].get('text')
                    if license_number:  # Only write if license number text is not None
                        f.write('{},{},{},{},{},{},{}\n'.format(
                            frame_nmr,
                            car_id,
                            '[{} {} {} {}]'.format(*car_data['car']['bbox']),
                            '[{} {} {} {}]'.format(*car_data['license_plate']['bbox']),
                            car_data['license_plate']['bbox_score'],
                            license_number,
                            car_data['license_plate']['text_score'] or "None"
                        ))

def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if 5 <= len(text) <= 7 and all(char in string.ascii_uppercase + "0123456789" for char in text):
        return True
    return False

def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in range(len(text)):
        if j in mapping and text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]
    return license_plate_

def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image with enhanced preprocessing and OCR fallback.

    Args:
        license_plate_crop (numpy.ndarray): Cropped image containing the license plate.

    Returns:
        tuple: Formatted license plate text and its confidence score, or (None, None) if no valid text.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
    
    # Resize for better OCR accuracy
    gray_resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray_resized, (3, 3), 0)
    
    # Sharpen image to make characters clearer
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(blurred, -1, kernel)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8)
    
    # Optional: Dilate to make characters more distinct
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    processed_image = cv2.dilate(thresh, kernel_dilate, iterations=1)
    
    # Debugging: Save the processed image for inspection
    debug_plate_path = "./debug_license_plate_processed.jpg"
    cv2.imwrite(debug_plate_path, processed_image)
    print(f"Saved processed license plate image for debugging at {debug_plate_path}")

    # OCR Configuration for pytesseract
    ocr_config = "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    
    # Attempt OCR with pytesseract
    license_plate_text = pytesseract.image_to_string(processed_image, config=ocr_config).strip()
    print("Detected License Plate Text (pytesseract):", license_plate_text)

    # Check if detected text meets the format requirements
    if license_plate_text and license_complies_format(license_plate_text):
        formatted_text = format_license(license_plate_text)
        print(f"Formatted license plate text (pytesseract): {formatted_text}")
        return formatted_text, 0.85  # Placeholder confidence score

    # Fallback to EasyOCR if pytesseract result is unsatisfactory
    print("Fallback to EasyOCR for OCR...")
    easyocr_text = read_license_plate_with_easyocr(license_plate_crop)

    if easyocr_text and license_complies_format(easyocr_text):
        formatted_text = format_license(easyocr_text)
        print(f"Formatted license plate text (EasyOCR): {formatted_text}")
        return formatted_text, 0.85  # Placeholder confidence score

    # If no valid text is found
    print("License plate text did not meet format requirements.")
    return None, None

def read_license_plate_with_easyocr(license_plate_crop):
    """
    Read the license plate text using EasyOCR as a fallback method.

    Args:
        license_plate_crop (numpy.ndarray): Cropped image containing the license plate.

    Returns:
        str: Detected license plate text, or None if no text was found.
    """
    result = reader.readtext(license_plate_crop, detail=0)  # Extract text only
    return result[0] if result else None

def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1
