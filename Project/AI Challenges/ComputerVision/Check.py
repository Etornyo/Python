import pandas as pd
import numpy as np
import cv2
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess data
def load_and_preprocess_data(csv_path, img_folder_path):
    df = pd.read_csv(csv_path)
    df_norm = df[['xmin', 'xmax', 'ymin', 'ymax']].copy()

    for index, row in df_norm.iterrows():
        image_path = img_folder_path + row['img_id']
        image = cv2.imread(image_path)
        h, w, _ = image.shape
        df_norm.loc[index] = df_norm.loc[index] / [w, w, h, h]

    df[['xmin', 'xmax', 'ymin', 'ymax']] = df_norm
    return df

# Split the dataset
def split_dataset(df, test_size=0.2, random_state=42):
    X = []
    y = []

    for index, row in df.iterrows():
        image_path = img_folder_path + row['img_id']
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))  # Resize image to 224x224
        image = image / 255.0  # Normalize image
        X.append(image)

        label = [row['xmin'], row['xmax'], row['ymin'], row['ymax']]
        y.append(label)

    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Define the model architecture
def build_model():
    model = Sequential()
    model.add(Conv2D(128, (3, 3), input_shape=(224, 224, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(8, activation="relu"))
    model.add(Dense(4, activation="sigmoid"))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

# Train the model
def train_model(model, X_train, y_train, X_val, y_val, batch_size=64, epochs=20):
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))
    return history

# Save predictions to CSV
def save_predictions(model, df, img_folder_path, output_csv_path):
    predictions = []
    for index, row in df.iterrows():
        image_path = img_folder_path + row['img_id']
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)[0]
        predictions.append(prediction)

    df[['xmin_pred', 'xmax_pred', 'ymin_pred', 'ymax_pred']] = predictions
    df.to_csv(output_csv_path, index=False)

# Define paths
csv_path = "/content/license_plates_detection_train.csv"
img_folder_path = "/content/license_plates_detection_train/"
output_csv_path = "predictions.csv"

# Load and preprocess data
df = load_and_preprocess_data(csv_path, img_folder_path)

# Split the dataset
X_train, X_test, y_train, y_test = split_dataset(df)

# Build and train the model
model = build_model()
history = train_model(model, X_train, y_train, X_test, y_test)

# Save predictions to CSV
save_predictions(model, df, img_folder_path, output_csv_path)
