import numpy as np
import pandas as pd
import cv2
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Data Preprocessing
def preprocess_data(df_detect):
    df_norm = pd.DataFrame(columns=['nxmin', 'nxmax', 'nymin', 'nymax'])
    for i in range(len(df_detect)):
        image_path = "/content/license_plates_detection_train/" + df_detect.iloc[i]["img_id"]
        img_arr = cv2.imread(image_path)
        h, w, d = img_arr.shape

        ymax, ymin, xmax, xmin = df_detect.iloc[i][["ymax", "ymin", "xmax", "xmin"]]
        df_norm.loc[len(df_norm)] = [xmin / w, xmax / w, ymin / h, ymax / h]

    df_detect = pd.concat([df_detect, df_norm], axis=1)
    return df_detect

# Model Creation
def build_model(base_model):
    base_model = base_model(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    head_model = base_model.output
    head_model = Flatten()(head_model)
    head_model = Dense(1024, activation="relu")(head_model)
    head_model = Dense(128, activation="relu")(head_model)
    head_model = Dense(4, activation='sigmoid')(head_model)

    model = Model(inputs=base_model.input, outputs=head_model)
    return model

# Data Loading
def load_data(df_detect):
    data = []
    labels = []

    for i in range(len(df_detect)):
        image_path = "/content/license_plates_detection_train/" + df_detect.iloc[i]["img_id"]
        img_arr = cv2.imread(image_path)
        h, w, _ = img_arr.shape

        load_image = load_img(image_path, target_size=(224, 224))
        load_image_arr = img_to_array(load_image)
        norm_load_image_arr = load_image_arr / 255.0

        xmin, xmax, ymin, ymax = df_detect.iloc[i][["xmin", "xmax", "ymin", "ymax"]]
        nxmin, nxmax = xmin / w, xmax / w
        nymin, nymax = ymin / h, ymax / h
        label_norm = (nxmin, nxmax, nymin, nymax)

        data.append(norm_load_image_arr)
        labels.append(label_norm)

    X = np.array(data, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    return X, y

# Model Training
def train_model(model, X_train, y_train, X_val, y_val):
    model.compile(loss='mse', optimizer=Adam(learning_rate=1e-4))
    callbacks = [EarlyStopping(patience=5, restore_best_weights=True), ReduceLROnPlateau(factor=0.1, patience=3)]
    history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val), callbacks=callbacks)
    return history

# Load and Preprocess Data
df_detect = pd.read_csv("/content/license_plates_detection_train.csv")
df_detect = df_detect[['img_id', 'xmin', 'xmax', 'ymin', 'ymax']]
df_detect = preprocess_data(df_detect)

# Load Data
X, y = load_data(df_detect)

# Split Data
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=0)

# Build Models
resnet_model = build_model(ResNet50)
vgg16_model = build_model(VGG16)

# Train Models
resnet_history = train_model(resnet_model, X_train, y_train, X_val, y_val)
vgg16_history = train_model(vgg16_model, X_train, y_train, X_val, y_val)

# Save Models
resnet_model.save('resnet_object_detection_best.h5')
vgg16_model.save('vgg16_object_detection_best.h5')
