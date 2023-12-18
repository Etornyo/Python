import cv2
import dlib
import numpy as np

def draw_rectangle(frame, point1, point2):
    cv2.rectangle(frame, point1, point2, color=(0, 255, 0), thickness=2)

def draw_face_emotion(face_emotion):
    text = "Neutral"
    color = (0, 255, 0)
    if face_emotion == 1:
        text = "Angry"
        color = (0, 0, 255)
    elif face_emotion == 2:
        text = "Disgusted"
        color = (0, 255, 255)
    elif face_emotion == 3:
        text = "Surprised"
        color = (255, 0, 0)
    elif face_emotion == 4:
        text = "Happy"
        color = (255, 255, 0)
    elif face_emotion == 5:
        text = "Sad"
        color = (255, 0, 255)
    return text, color

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
emotion_classifier = dlib.train_svm_predictor("svm_predictor.dat", "images.xml")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    faces = detector(frame)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        draw_rectangle(frame, (x1, y1), (x2, y2))
        landmarks = predictor(frame, face)
        face_emotion = emotion_classifier.predict(landmarks)
        face_emotion_text, face_emotion_color = draw_face_emotion(face_emotion)
        cv2.putText(frame, face_emotion_text, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, face_emotion_color, 2)
    cv2.imshow("Emotion Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()