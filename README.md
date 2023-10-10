# Face-Recognition
We use the [haarcascade_frontalface_default.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml) for implementing this.\
Please refer [haarcascades](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html).\
Save the xml file in your cwd.

## Dataset_Trainer.py
Captures 100 images and outputs a yaml file (custom face cascade model).
  ```py
  face_cascade = cv2.CascadeClassifier("path/to/haarcascade_frontalface_default.xml")
  face_recognizer = cv2.face.LBPHFaceRecognizer_create()
  ```

## main.py
Iterates through all the available yaml files in your locale and predicts the input using nearest neighbour algorithm.

```py
label, confidence = face_recognizer.predict(predicted face region)
```

## front-end.py
Integration of CV2 window with tkinter for more window controls. (Under Development)
