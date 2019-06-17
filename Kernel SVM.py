# Kernel SVM accuracy 0.75

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

# Importing the dataset

path = r'C:\Users\USER\Desktop\Future Project\AI for SEA\safety\features' 
all_files = glob.glob(os.path.join(path, "part*.csv"))     

df_from_each_file = (pd.read_csv(f) for f in all_files)
dataraw   = pd.concat(df_from_each_file, ignore_index=True)

datamean = dataraw.groupby(['bookingID']).agg({ 'Accuracy': 'mean', 
                                                'Bearing': 'mean',
                                                'acceleration_x': 'mean',
                                                'acceleration_y': 'mean',
                                                'acceleration_z': 'mean',
                                                'gyro_x': 'mean',
                                                'gyro_y': 'mean',
                                                'gyro_z': 'mean',
                                                'second': 'mean',
                                                'Speed': 'mean',
                                                }).reset_index()

datalabel = pd.read_csv(r"C:\Users\USER\Desktop\Future Project\AI for SEA\safety\labels\part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv")
dataset = pd.merge(datamean, datalabel, how='inner', on=['bookingID'])


X = dataset.iloc[:, 1:11].values
y = dataset.iloc[:, 11].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

