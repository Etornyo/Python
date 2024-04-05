import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
import cv2
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load pre-trained YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Define XGBoost model
xgb_model = xgb.XGBRegressor()

# Load and preprocess data
df_detect = pd.read_csv("license_plates_detection_train.csv")

# Define function to preprocess image for YOLO
def preprocess_image_yolo(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = np.array(img)
    return img[:, :, ::-1].copy()

# Define function to preprocess image for XGBoost
def preprocess_image_xgb(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize((224, 224))  # Resize image to match YOLO input size
    img = np.array(img)
    return img

# Make predictions using YOLO and preprocess data for XGBoost
data = []
labels = []
for i in range(len(df_detect)):
    image_path = "license_plates_detection_train" + df_detect.iloc[i]["img_id"]
    yolo_image = preprocess_image_yolo(image_path)
    results = model(yolo_image)  # Get YOLO predictions
    prediction = results.xyxy[0].cpu().numpy()  # Convert predictions to numpy array

    # Extract bounding box coordinates and other features from YOLO predictions
    xmin, ymin, xmax, ymax, confidence, class_id = prediction[0]  # Assuming only one prediction per image
    features = [xmin, ymin, xmax, ymax, confidence]  # Example features

    # Preprocess image for XGBoost
    xgb_image = preprocess_image_xgb(image_path)

    data.append(xgb_image)
    labels.append(features)

# Convert lists to numpy arrays
X = np.array(data)
y = np.array(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
xgb_model.fit(X_train, y_train)

# Make predictions using XGBoost model
y_pred = xgb_model.predict(X_test)

# Evaluate XGBoost model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Save trained XGBoost model
xgb_model.save_model('xgb_object_detection.model')


