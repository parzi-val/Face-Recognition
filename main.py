import cv2
import os
import csv

with open("data.csv", "r") as f:
    reader = csv.reader(f)
    temp = {row[1]: int(row[0]) for row in reader if row}

dataset = {v: k for k, v in temp.items()}
print(dataset)

# Initialize the cascade classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Initialize the camera
cap = cv2.VideoCapture(0)

# Load the trained LBPHFaceRecognizer for each person
face_recognizers = {}
for person_yaml in os.listdir("faces"):
    print(person_yaml)
    person_id = temp[person_yaml[:-5]]
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(os.path.join("faces", person_yaml))
    face_recognizers[person_id] = recognizer

print(face_recognizers)

# Load the dataset to map label ids to person names


while True:
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
                print(predicted_person)

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

    # Display the resulting frame
    cv2.imshow("Video", frame)

    # Exit the program if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
