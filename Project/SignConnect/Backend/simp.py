import cv2
import numpy as np
import os
from matplotlib  import pyplot as plt
import time
import mediapipe as mp



mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils



def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #color conversion
    image.flags.writeable = False #set the image flag to not writeable  
    results = model.process(image) # Make predictions/run inference on the image  
    image.flags.writeable = True #m ake the image flags writable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # change color back from RGB to BGR  
    
    return image, results
    
    #draw landmarks on
    
    
    
    

def draw_styled_landmarks(image, results):
    # Draw face Mesh connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),# BGR for colors
                            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    
    # Draw Pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
                            ,mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=4),
                            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=2)
                            )
    
    
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
                            ,mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                            )
    
    
    # Draw Right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
                            , mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                            )
    
    
    
    
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        
        # Read feed
        ret, frame = cap.read()
        
        # Make detaction
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        
        # Draw landmarks
        # draw_landmarks(image, results)
        draw_styled_landmarks(image , results)
        
        # Show to screen
        cv2.imshow('SignAI', image)
        
        # Break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
    
# draw_landmarks(frame, results)
draw_styled_landmarks(frame,results)

plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def extract_keypoints(results):
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    left = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    right = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([face, pose, left, right])


extract_keypoints(results).shape