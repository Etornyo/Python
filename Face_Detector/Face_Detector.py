import cv2


# loading pre-trained data on frontals
train_face_data = cv2.CascadeClassifier('C:\\Users\\etord\\OneDrive\\Desktop\\ME\\Git\Python\\Face_Detector\\haarcascade_frontalface_default.xml') # better safeto use paths 
#CascadeClassifier is a detector for classification of you know

# Choosing an image to detect faces but you must specify the path unless on Desktop
img = cv2.imread('C:\\Users\\etord\\OneDrive\\Desktop\\ME\Git\\Python\\Face_Detector\\1829569.jpg')# change images here



# Check if the image data is loaded correctly
if img is None:
    print("Error loading image.")
else:
    print("Image loaded successfully.")


# changing to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Detect faces
face_coordinates = train_face_data.detectMultiScale(grayscaled_img)


# Draw rectangles
for (x,y,w,h) in face_coordinates: # (x,y,w,h) = face_coordinates[0] when iterating once. use miultiples to iterate further
    cv2.rectangle(img,(x,y),(x+w,y+h),(221,255,7),3)

# print(face_coordinates)

#img2 = cv2.imread('3.jpeg')
#img3 = cv2.imread('4.jpeg')
#img4 = cv2.imread('5.jpeg')

# '' marks determine the name of the tab that pops up followed by the variable for the image
cv2.imshow('Face_Detector',img ) 

# Pauses execution of code until you press a button then it closes
cv2.waitKey(0) 
#cv2.imshow('Face_Detector',img2) # '' marks determine the name of the tab that pops up followed by the variable for the image
#cv2.waitKey()
#cv2.imshow('Face_Detector',img3) # '' marks determine the name of the tab that pops up followed by the variable for the image
#cv2.waitKey()
#cv2.imshow('Face_Detector',img4) # '' marks determine the name of the tab that pops up followed by the variable for the image
#cv2.waitKey()




print("Code Done") 