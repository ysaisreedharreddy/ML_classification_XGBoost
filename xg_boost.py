# Importing necessary libraries for data manipulation and visualization
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r"C:\Users\prasu\DS2\git\classification\7. XGBOOST\Churn_Modelling.csv")
X = dataset.iloc[:, 3:-1].values  # Selecting relevant feature columns
y = dataset.iloc[:, -1].values    # Selecting the target column

# Display initial data arrays for verification
print(X)
print(y)

# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])  # Transform 'Gender' column into numerical format
print(X)

# One Hot Encoding the "Geography" column to avoid ordinal implications
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))  # Apply OneHotEncoder to 'Geography' and transform the result to a numpy array
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training XGBoost on the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.0001) 
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix to evaluate the predictions
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Calculating the accuracy of the model
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print('Accuracy:', ac)

# Display model's training and test set performance
bias = classifier.score(X_train, y_train)  # Model's accuracy on the training set
variance = classifier.score(X_test, y_test)  # Model's accuracy on the test set
print('Bias (Training accuracy):', bias)
print('Variance (Test accuracy):', variance)

# Applying k-Fold Cross Validation to evaluate model's performance
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print("Accuracy: {:.2f} %".format(accuracies.mean() * 100))
#print("Standard Deviation: {:.2f} %".format(accuracies.std() * 100))
