import cv2
import os
import numpy as np
import csv

# create a face cascade classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# initialize the LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Create the faces directory if it doesn't exist
if not os.path.exists("faceImages"):
    os.mkdir("faceImages")
if not os.path.exists("faces"):
    os.mkdir("faces")

# Get the name of the person to train
name = input("Enter your name: ")

# Create the directory for the person's images
person_dir = os.path.join("faceImages", name)
if not os.path.exists(person_dir):
    os.mkdir(person_dir)

# initialize the camera
cap = cv2.VideoCapture(0)

# capture 50 images of the user's face
count = 0

while count < 100:
    ret, frame = cap.read()
    if ret:
        # convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect the face
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # save the face images
        for (x, y, w, h) in faces:
            cv2.imwrite(
                f"faceImages/{name}/{name}_{count}.jpg", gray[y : y + h, x : x + w]
            )
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            count += 1

        # display the video feed
        cv2.imshow("frame", frame)

        # press 'q' to exit
        if cv2.waitKey(1) == ord("q"):
            break

# release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

index = len(os.listdir("faces"))


# iterate over the saved images to train the model
images = []
labels = []
for root, dirs, files in os.walk(f"faceImages/{name}"):
    for file in files:
        if file.endswith("jpg"):
            img_path = os.path.join(root, file)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            images.append(image)
            labels.append(index)  # assign the same label to all images of the user


print(labels)

with open("data.csv", "a") as dataset:
    writer = csv.writer(dataset)
    writer.writerow([index, name])

# train the model on the new images
face_recognizer.train(images, np.array(labels))

# save the new trained yaml file
face_recognizer.save(f"faces/{name}.yaml")

# delete the previous yaml file if it exists
if os.path.exists("faces.yaml"):
    os.remove("faces.yaml")

# create a new yaml file with all the saved faces
face_files = [
    f"faces/{f}"
    for f in os.listdir("faces")
    if f.endswith(".yaml") and f != f"{name}.yaml"
]
