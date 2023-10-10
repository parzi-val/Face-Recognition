import cv2
import os
import csv
from tkinter import *
from PIL import Image, ImageTk


def destructor():
    root.destroy()
    cap.release()


with open("data.csv", "r") as f:
    reader = csv.reader(f)
    temp = {row[1]: int(row[0]) for row in reader if row}

dataset = {v: k for k, v in temp.items()}


# Initialize the cascade classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Initialize the camera
cap = cv2.VideoCapture(0)

# Load the trained LBPHFaceRecognizer for each person
face_recognizers = {}
for person_yaml in os.listdir("faces"):
    person_id = temp[person_yaml[:-5]]
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(os.path.join("faces", person_yaml))
    face_recognizers[person_id] = recognizer


def video_loop():
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Crop the face region from the grayscale frame
        roi_gray = gray[y : y + h, x : x + w]

        # Recognize the face using the trained LBPHFaceRecognizer for each person
        predicted_person = None
        min_distance = 60
        for person_id, recognizer in face_recognizers.items():
            label, confidence = recognizer.predict(roi_gray)
            if confidence < min_distance:
                min_distance = confidence
                predicted_person = person_id

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the predicted label and confidence score
        name = dataset.get(predicted_person, "Unknown")
        cv2.putText(
            frame,
            "Name: " + str(name),
            (x, y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "Confidence: " + str(min_distance),
            (x, y + h + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=current_image)
            panel.imgtk = imgtk
            panel.config(image=imgtk)
    root.after(10, video_loop)


root = Tk()
root.geometry("700x500")
root.protocol("WM_DELETE_WINDOW", destructor)
panel = Label(root)
panel.pack(padx=10, pady=10)
video_loop()
root.mainloop()
