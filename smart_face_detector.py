import cv2
import face_recognition
import os
import numpy as np
from deepface import DeepFace

# --- 1. Load Known Faces ---
# This part scans the 'known_faces' directory and learns each person's face.

print("Loading known faces...")
known_face_encodings = []
known_face_names = []

# The path to the directory containing sub-folders of known people.
KNOWN_FACES_DIR = "known_faces"

# Loop through each person's folder in the known_faces directory.
for name in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    
    # Check if the path is a directory before processing.
    if os.path.isdir(person_dir):
        # Loop through each image of the person.
        for filename in os.listdir(person_dir):
            image_path = os.path.join(person_dir, filename)
            
            # Load the image file.
            image = face_recognition.load_image_file(image_path)
            
            # Get the face encoding (a unique "fingerprint" for the face).
            # We assume each photo has only one face.
            face_encodings = face_recognition.face_encodings(image)
            
            if face_encodings:
                encoding = face_encodings[0]
                # Add the encoding and the person's name to our lists.
                known_face_encodings.append(encoding)
                known_face_names.append(name)

print(f"Loaded {len(known_face_names)} known faces.")

# --- 2. Initialize Webcam and Variables ---

# Use camera index 0 for the default webcam.
video_capture = cv2.VideoCapture(0) 

# Check if the webcam is opened correctly.
if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started. Looking for faces...")

# --- 3. The Main Loop (Real-time Detection) ---
# This loop continuously grabs frames from the webcam and processes them.

while True:
    # Grab a single frame of video.
    ret, frame = video_capture.read()
    if not ret:
        break # Exit if there's an issue reading from the webcam.

    # Find all face locations and encodings in the current frame.
    # This is more efficient than processing each face one by one.
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face found in the frame.
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        
        # --- Person Recognition ---
        # Compare the found face to all known faces.
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Person" # Default name if no match is found.

        # Find the best match among the known faces.
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        # --- Emotion Recognition ---
        emotion = "Detecting..."
        try:
            # The DeepFace.analyze function can detect emotion, age, gender, etc.
            # We tell it to only perform emotion analysis to save time.
            analysis = DeepFace.analyze(
                frame, 
                actions=['emotion'], 
                enforce_detection=False # Don't re-detect the face, use the one we found.
            )
            # The result is a list, so we get the first item.
            # The dominant emotion is stored in 'dominant_emotion'.
            emotion = analysis[0]['dominant_emotion']
        except Exception as e:
            # If DeepFace fails (e.g., face is too blurry or small), we just skip it.
            emotion = "N/A"

        # --- Draw Results on the Frame ---
        # Draw a green box around the face.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Create the label with the person's name and their emotion.
        label = f"{name} ({emotion})"
        
        # Draw a filled rectangle for the label's background.
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        
        # Write the text on the background.
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, label, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image in a window.
    cv2.imshow('Smart Face Detector', frame)

    # Hit 'q' on the keyboard to quit the program.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 4. Cleanup ---
# Release handle to the webcam and close all windows.
video_capture.release()
cv2.destroyAllWindows()
print("Application closed.")
