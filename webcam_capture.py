# -----------------------------------------------------------------------------
#                                   webcam_capture.py
# =============================================================================
#  Captures images of individuals detected by a webcam in real-time. 
#
#
#   Usage:   pip3 install opencv-python
#            python webcam_capture.py
#
#   Last updated: 24 Oct 2024 by mctrinh
# -----------------------------------------------------------------------------

import cv2      
import time
import os

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load the pre-trained Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create directory for saved images if it doesn't exist
save_dir = 'D:/code/webcam-capture/'
os.makedirs(save_dir, exist_ok=True)

# Set a capture interval in seconds
capture_interval = 2
last_capture_time = time.time()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Check if enough time has passed to capture an image
        current_time = time.time()
        if current_time - last_capture_time >= capture_interval:
            # Save the captured image
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(save_dir, f"{timestamp}.jpg")
            cv2.imwrite(file_path, frame)
            print(f"Image {timestamp} captured and saved!")
            last_capture_time = current_time

    # Display the resulting frame
    cv2.imshow('Webcam', frame)

    # Break the loop on 'ESC' key press
    if cv2.waitKey(30) & 0xFF == 27:
        print("ESC key pressed. Exiting...")
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
