import dlib
import cv2

# Load the 2D image
image = cv2.imread ( 'input_image.jpg' )

# Convert the image to grayscale
gray_image = cv2.cvtColor ( image, cv2.COLOR_BGR2GRAY )

# Initialize the Multi-PIE 2D 106 facial landmark detector
detector = dlib.get_frontal_face_detector ()
predictor = dlib.shape_predictor ( 'shape_predictor_68_face_landmarks.dat' )

# Detect faces in the image
faces = detector ( gray_image )

# For each detected face, get the facial landmarks
for face in faces:
    landmarks = predictor ( gray_image, face )

 