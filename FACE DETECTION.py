import cv2
import os
import time
from datetime import datetime

# Create a directory to store captured images
CAPTURE_DIR = "captured_images"
os.makedirs(CAPTURE_DIR, exist_ok=True)

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
video_capture = cv2.VideoCapture(0)  # Adjust the index if the webcam is not recognized

if not video_capture.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Starting video stream...")

try:
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # Convert the frame to grayscale (required for face detection)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around detected faces and capture images
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Capture and save the face
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            face_image = frame[y:y+h, x:x+w]
            face_filename = os.path.join(CAPTURE_DIR, f"face_{timestamp}.png")
            cv2.imwrite(face_filename, face_image)
            print(f"Captured and saved: {face_filename}")

        # Display the resulting frame
        cv2.imshow('Video Stream', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting video stream...")
            break

except KeyboardInterrupt:
    print("Interrupted. Exiting...")

finally:
    # Release the webcam and close OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()
    print("Resources released and windows closed.")
