import pandas as pd
import numpy as np


wk = pd.read_csv('C:\Users\etord\OneDrive\Desktop\ME\Git\Python\Project\AI Challenges\Fraud\client_train.csv')
wk.head()


# Isolation
from sklearn.ensemble import IsolationForest

# Assuming 'X' is your feature matrix
model_iforest = IsolationForest(contamination=0.05)  # Adjust contamination based on your dataset
model_iforest.fit(X)

# Predicting anomalies (1 for normal, -1 for anomaly)
predictions_iforest = model_iforest.predict(X)





# Local Outlier Factor
from sklearn.neighbors import LocalOutlierFactor

# Assuming 'X' is your feature matrix
model_lof = LocalOutlierFactor(contamination=0.05)  # Adjust contamination based on your dataset
predictions_lof = model_lof.fit_predict(X)



# One-class SVM
from sklearn.svm import OneClassSVM

# Assuming 'X' is your feature matrix
model_ocsvm = OneClassSVM(nu=0.05)  # Adjust nu based on your dataset
predictions_ocsvm = model_ocsvm.fit_predict(X)



# Autoencoders using Tensorflow
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler

# Assuming 'X' is your feature matrix
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

input_dim = X_scaled.shape[1]
encoding_dim = 10  # You can experiment with different values

# Define the autoencoder model
input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="relu")(input_layer)
decoder = Dense(input_dim, activation="sigmoid")(encoder)
autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoder)

# Compile and fit the autoencoder
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.fit(X_scaled, X_scaled, epochs=10, batch_size=32, shuffle=True, validation_split=0.2)

# Use reconstruction errors as anomaly scores
reconstructions = autoencoder.predict(X_scaled)
mse = ((X_scaled - reconstructions) ** 2).mean(axis=1)
threshold_autoencoder = mse.mean() + 3 * mse.std()  # Adjust the threshold based on your dataset

# Predict anomalies (1 for normal, -1 for anomaly)
predictions_autoencoder = (mse > threshold_autoencoder).astype(int)


# Train split
from sklearn.model_selection import train_test_split

# Assuming 'X' is your feature matrix and 'y' is the target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

