import cv2
import os
import numpy as np
import serial
import time

# Set up serial communication (Change 'COM4' to your Arduino port)
ser = serial.Serial('COM4', 9600, timeout=1)
time.sleep(2)  # Allow time for the connection to establish

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create a folder to save detected faces if it doesn't exist
output_folder = "Detected_faces"
os.makedirs(output_folder, exist_ok=True)

# Function to check if a new face is unique
def is_similar_face(new_face, existing_face_path):
    existing_face = cv2.imread(existing_face_path, cv2.IMREAD_GRAYSCALE)
    existing_face = cv2.resize(existing_face, (new_face.shape[1], new_face.shape[0]))

    hist_new = cv2.calcHist([new_face], [0], None, [256], [0, 256])
    hist_existing = cv2.calcHist([existing_face], [0], None, [256], [0, 256])

    cv2.normalize(hist_new, hist_new, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_existing, hist_existing, 0, 1, cv2.NORM_MINMAX)

    similarity = cv2.compareHist(hist_new, hist_existing, cv2.HISTCMP_CORREL)
    return similarity > 0.8  # Higher similarity means duplicate face

def is_new_face(new_face):
    for filename in os.listdir(output_folder):
        existing_face_path = os.path.join(output_folder, filename)
        if is_similar_face(new_face, existing_face_path):
            os.remove(existing_face_path)
            print(f"Removed duplicate: {existing_face_path}")
            return False  # Not a new face
    return True

# Variables to manage camera and passkey system
camera_active = False
webcam = None
passkey_required = False  # Initially no passkey is needed
registration_count = 0  # Count of face registrations
stored_passkey = None  # To store the passkey after first registration

while True:
    if ser.in_waiting > 0:
        command = ser.readline().decode().strip()
        if command == "open_camera" and not camera_active and not passkey_required:
            print("Camera opening... Waiting for face registration.")
            stored_passkey = input('Setup your PASSKEY for the first registration: ')  # Store the passkey
            camera_active = True
            webcam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
            webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Lower width
            webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Lower height
            webcam.set(cv2.CAP_PROP_XI_FRAMERATE,120)
            time.sleep(2)

        elif command == stored_passkey and passkey_required:
            print("Passkey received! You can register the next face.")
            passkey_required = False  # Reset passkey requirement
            camera_active = True
            webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            time.sleep(2)
            webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        elif command == "stop_led":
            print("Stopping camera...")
            camera_active = False
            if webcam:
                webcam.release()
            cv2.destroyAllWindows()

    if camera_active and webcam and webcam.isOpened():
        time.sleep(0.1)  # Slight delay to avoid rapid loop execution
        ret, img = webcam.read()
        if not ret:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 1:
            x, y, w, h = faces[0]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            face = gray[y:y+h, x:x+w]

            if is_new_face(face):
                face_filename = os.path.join(output_folder, f"face_{len(os.listdir(output_folder)) + 1}.jpg")
                cv2.imwrite(face_filename, face)
                print(f"New face saved as {face_filename}")
                registration_count += 1
                if registration_count == 1:
                    choice = input("Do you want to register another face? (yes/no): ")
                    if choice.lower() == "yes":
                        print("Please enter the passkey to register another face.")
                        passkey_required = True
                    else:
                        print("Face registration complete.")
                        camera_active = False
                        webcam.release()
                        cv2.destroyAllWindows()
                else:
                    print(f"Face {registration_count} registered. You can stop or add more faces.")

                camera_active = False
                webcam.release()
                cv2.destroyAllWindows()

        cv2.imshow("FACE REGISTERED", img)
        if cv2.waitKey(10) == 27:  # Press 'Esc' to exit manually
            break
        time.sleep(0.5)  # Avoid rapid re-capture

# Cleanup before exit
if webcam:
    webcam.release()
cv2.destroyAllWindows()
ser.close()
