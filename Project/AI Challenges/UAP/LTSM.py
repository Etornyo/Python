import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

# Load datasets
train_data = pd.read_csv('Train.csv')
test_data = pd.read_csv('Test.csv')

# Data preprocessing
train_data.fillna(0, inplace=True)
test_data.fillna(0, inplace=True)

# Feature selection
features = ['precipitable_water_entire_atmosphere',
       'relative_humidity_2m_above_ground',
       'specific_humidity_2m_above_ground', 'temperature_2m_above_ground',
       'u_component_of_wind_10m_above_ground',
       'v_component_of_wind_10m_above_ground',
       'L3_NO2_NO2_column_number_density',
       'L3_NO2_NO2_slant_column_number_density',
       'L3_NO2_absorbing_aerosol_index', 'L3_NO2_cloud_fraction',
       'L3_NO2_sensor_altitude',  'L3_NO2_solar_azimuth_angle',
       'L3_NO2_solar_zenith_angle',
       'L3_NO2_stratospheric_NO2_column_number_density',
       'L3_NO2_tropopause_pressure',
       'L3_NO2_tropospheric_NO2_column_number_density',
       'L3_O3_O3_column_number_density', 'L3_O3_O3_effective_temperature',
       'L3_O3_cloud_fraction', 'L3_O3_solar_azimuth_angle',
       'L3_O3_solar_zenith_angle', 'L3_CO_CO_column_number_density',
       'L3_CO_H2O_column_number_density', 'L3_CO_cloud_height',
       'L3_CO_sensor_altitude', 'L3_CO_solar_azimuth_angle',
       'L3_CO_solar_zenith_angle', 'L3_HCHO_HCHO_slant_column_number_density',
       'L3_HCHO_cloud_fraction',  'L3_HCHO_solar_azimuth_angle',
       'L3_HCHO_solar_zenith_angle',
       'L3_HCHO_tropospheric_HCHO_column_number_density',
       'L3_HCHO_tropospheric_HCHO_column_number_density_amf',
       'L3_CLOUD_cloud_base_height', 'L3_CLOUD_cloud_base_pressure',
       'L3_CLOUD_cloud_fraction', 'L3_CLOUD_cloud_optical_depth',
       'L3_CLOUD_cloud_top_height', 'L3_CLOUD_cloud_top_pressure',
       'L3_CLOUD_solar_azimuth_angle', 'L3_CLOUD_solar_zenith_angle',
       'L3_CLOUD_surface_albedo', 'L3_AER_AI_absorbing_aerosol_index',
       'L3_AER_AI_sensor_altitude',
       'L3_AER_AI_solar_azimuth_angle',
       'L3_AER_AI_solar_zenith_angle', 'L3_SO2_SO2_column_number_density',
       'L3_SO2_SO2_column_number_density_amf',
       'L3_SO2_SO2_slant_column_number_density',
       'L3_SO2_absorbing_aerosol_index', 'L3_SO2_cloud_fraction',
       'L3_SO2_solar_azimuth_angle', 'L3_SO2_solar_zenith_angle',
       'L3_CH4_aerosol_height', 'L3_CH4_aerosol_optical_depth',
       'L3_CH4_solar_azimuth_angle', 'L3_CH4_solar_zenith_angle']  # Define your features

# Split data
X = train_data[features]
y = train_data['target']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data for LSTM input (assuming X_train and X_valid are numpy arrays)
X_train = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1])
X_valid = X_valid.values.reshape(X_valid.shape[0], 1, X_valid.shape[1])

# Define LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))

# Compile model
optimizer = Adam(lr=0.001)
model.compile(loss='mse', optimizer=optimizer)

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_valid, y_valid), verbose=1)

# Model evaluation
lstm_predictions = model.predict(X_valid)
lstm_mse = mean_squared_error(y_valid, lstm_predictions)

# Model prediction on test dataset
test_predictions_lstm = model.predict(test_data[features].values.reshape(test_data.shape[0], 1, test_data.shape[1]))

# Prepare submission file
submission = pd.DataFrame({
    'Place_ID X Date': test_data['Place_ID'] + ' X ' + test_data['Date'].astype(str),
    'target': test_predictions_lstm.flatten()  # Flatten predictions from 3D to 1D
})
submission.to_csv('submission_lstm.csv', index=False)
