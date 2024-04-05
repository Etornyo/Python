# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



tr_client = pd.read_csv('client_train.csv')
tr_invoice = pd.read_csv('invoice_train.csv')
ts_client = pd.read_csv('client_test.csv')
ts_invoice = pd.read_csv('invoice_test.csv')



#Manipulation
tr_client.head()
tr_invoice.head()

tr_client.info()
tr_invoice.info()

tr_client.describe()
tr_invoice.describe()




#

# Assuming 'x_train', 'y_train', and 'test' are your training features, target variable, and test dataset

# Split the training data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
rf_classifier.fit(x_train, y_train)

# Make predictions on the validation set
val_predictions = rf_classifier.predict(x_val)

# Evaluate the accuracy on the validation set
accuracy = accuracy_score(y_val, val_predictions)
print(f"Validation Accuracy: {accuracy}")

# Make predictions on the test set
test_predictions = rf_classifier.predict(test)

# Create a DataFrame for submission
submit = pd.DataFrame({'client_id': sub_client_id, 'target': test_predictions})

# Save the submission file
submit.to_csv('submission_rf.csv', index=False)
