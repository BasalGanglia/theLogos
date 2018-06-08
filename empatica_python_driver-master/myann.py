# -*- coding: utf-8 -*-
"""
Created on Mon May  7 08:09:28 2018

@author: Ilkka
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# import keras:
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

##### STEPS TO TRAIN ANN WITH SGD

"""
STEP1: Randomly initialize the weights to small numbers close to 0
STEP2: Input the first observation of your dataset in the input layer,
       each feature in one input node.
STEP3: Forward-Propagation: from left to right, the neurons are activated in a way
       that the impact of each neuron's activation is limited by the weights. Propagate
       the activations until getting the predicted result y.
STEP4: Compare the predicted result to the actual result. Measure the generated error
STEP5: Back-Propagation: from right to left, the error is back-propagated. Update the weights
       according to how much they are responsible for the error. The laerning rate decides by how
       much we update the weights.
STEP6: Repated steps 1 to 5 and udpate the weights
STEP7: When the whole training set passes through the ANN we have one epoch. redo more epochs
"""

# ADDing the input and first hiddne alyer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Fitting classifier to the Training set
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)

# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)