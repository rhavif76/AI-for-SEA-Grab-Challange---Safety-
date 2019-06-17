# Kernel SVM 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

# Importing the dataset
# 1.1 import data feature and concatenate the data feature from Grab
path = r'C:\Users\USER\Desktop\uji coba ai\safety v2\features' 
all_files = glob.glob(os.path.join(path, "part*.csv"))     

df_from_each_file = (pd.read_csv(f) for f in all_files)
dataraw   = pd.concat(df_from_each_file, ignore_index=True)

#aggregate all data (mean)
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

# feature engineering the data, I think the most important data are acceleration and speed, i dont
# i dont know how to use the acccuracy
#  I made an assumption that this data get from smartphone that consumers use
# because of that the bearing data im not use because its depend of the smartphone position so the bearing data not really show the actual position of vehicle 
# and its also happen in gyro data because the data get from smartphone that depend on the hand position of consumers smartphone
# so i only use acceleration and speed in this Machine learning 
# the acceleration resultant function is (a^2+b^2+c^2)^0.5 
def fabc(a,b,c):
    return((a**2)+(b**2)+(c**2))**(0.5)
datamean['acceleration'] = datamean.apply(lambda x: fabc(x['acceleration_x'], x['acceleration_y'], x['acceleration_z']), axis=1)

# i combine the feature data and label data
datalabel = pd.read_csv(r"C:\Users\USER\Desktop\uji coba ai\safety v2\labels\part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv")
dataset = pd.merge(datamean, datalabel, how='inner', on=['bookingID'])

# the x and y data
X = dataset.iloc[:, [0,10,11]].values
y = dataset.iloc[:, 12].values



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

# the accuracy I get is 0.75 from confusion matrix using Kernel SVM