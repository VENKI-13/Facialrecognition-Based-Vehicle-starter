import cv2
import os
import numpy as np
import serial
import time
from twilio.rest import Client
import cloudinary
import cloudinary.uploader
import cloudinary.api

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create a folder to save detected faces if it doesn't exist
output_folder = "Detected_faces"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Set up serial communication
arduino = serial.Serial('COM4', 9600, timeout=1)
time.sleep(0.1)

# Twilio setup (replace with your actual Twilio credentials)
account_sid = 'zzzz'
auth_token = 'zzzz'
twilio_phone_number = '+zzz'
client = Client(account_sid, auth_token)
# Set up your Cloudinary credentials
cloudinary.config(
    cloud_name="yyy",
    api_key="yy",
    api_secret="yyy"
)

def upload_to_cloudinary(face_filename):
        response = cloudinary.uploader.upload(face_filename)
        return response['secure_url']

# SMS alert function
def send_sms_alert(face_filename):
    message = f"🚨 ALERT: Unknown face detected! Image saved as {upload_to_cloudinary(face_filename)}."
    to_phone_number = '+91xxxxx'  # Replace with the recipient's phone number

    # Send SMS
    client.messages.create(
        body=message,
        from_=twilio_phone_number,
        to=to_phone_number  # Recipient's phone number
    )
    print("SMS alert sent!")

# Train the recognizer
def train_recognizer():
    faces, labels = [], []
    for filename in os.listdir(output_folder):
        img = cv2.imread(os.path.join(output_folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            faces.append(img)
            labels.append(int(filename.split('_')[1].split('.')[0]))  # Assuming file name is face_X.jpg
        else:
            print(f"Warning: Couldn't load {filename}")
    if len(faces) > 0:
        recognizer.train(faces, np.array(labels))
        print("Model trained successfully!")
    else:
        print("No faces found to train the model.")
def preprocess_face(face):
    # Resize to standard dimensions (100x100)
    face_resized = cv2.resize(face, (100, 100))

    # Apply Gaussian blur to reduce noise
    face_blurred = cv2.GaussianBlur(face_resized, (3, 3), 0)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    face_equalized = clahe.apply(face_blurred)

    return face_equalized

def compare_with_saved_faces(live_face):
    # Preprocess the face before recognition
    live_face_processed = preprocess_face(live_face)

    # Predict the label and confidence of the live face
    label, confidence = recognizer.predict(live_face_processed)

    print(f'Confidence: {confidence}')

# Compare the live face with saved faces
def compare_with_saved_faces(live_face):
    # Resize the face to the expected size (e.g., 100x100)
    live_face_resized = cv2.resize(live_face, (100, 100))
    
    # Predict the label and confidence of the live face
    label, confidence = recognizer.predict(live_face_resized)
    print('Confidence: ', confidence)
    return (True, label) if confidence < 111 else (False, None)
camera_active = False

# Train the recognizer before starting the main loop
train_recognizer()

while True:
    if arduino.in_waiting > 0:
        command = arduino.readline().decode('utf-8', errors='ignore').rstrip()

        if command == "open_camera" and not camera_active:
            print("Opening Camera...")
            webcam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
            webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Lower width
            webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Lower height
            webcam.set(cv2.CAP_PROP_XI_FRAMERATE,120)   # Higher FPS
            camera_active = True

            while camera_active:
                ret, img = webcam.read()
                if not ret:
                    break

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

                if len(faces) == 1:
                    x, y, w, h = faces[0]
                    face = gray[y:y + h, x:x + w]

                    match_found, label = compare_with_saved_faces(face)
                    if match_found:
                        arduino.write(b"start\n")  # Turn LED on
                        cv2.putText(img, f"Match Found: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        print('Match found')
                        cv2.imshow("FACE DETECTION", img)
                        cv2.waitKey(2000)
                        break
                    else:
                        # Save the unrecognized face
                        no_match_filename = os.path.join(output_folder, f"unknown_{int(time.time())}.jpg")
                        cv2.imwrite(no_match_filename, face)
                        print(f"No Match found. Saved as {no_match_filename}")

                        # Send SMS alert
                        send_sms_alert(no_match_filename)

                        cv2.putText(img, "No Match", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.imshow("FACE DETECTION", img)
                if cv2.waitKey(10) == 27:  # Press 'Esc' to exit
                    break

            webcam.release()
            cv2.destroyAllWindows()
            camera_active = False  # Close camera after one recognition attempt

        elif command == "stop_led":
            print("Turning off Motor...")
            arduino.write(b"stop\n")
