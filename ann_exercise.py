# Artificial Neural Network

# Installing Theano
# conda install theano

# Installing Tensorflow
# conda install tensorflow

# Installing Keras
# conda install -c conda-forge keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3: 13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) #convert country to categorical variable
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2]) #convert gender to categorical variable
onehotencoder = OneHotEncoder(categorical_features = [1]) # create dummy variable for the country column (as there is no different of numbers for each category)
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] #remove the first column of dummy variable to avoid dummy variable trap


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) #in new python version, cross_validation is replaced by model_selection


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential #import neural networks
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer='glorot_uniform', activation = 'relu', input_dim = 11))
#units is the nodes of the hidden layer - usually its the average of input variables and outputs
#input_dim is the number of input variables; 'relu' represents rectifier activiation function

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer='glorot_uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer='glorot_uniform', activation = 'sigmoid'))
# units is the number of output variables - 1; use sigmoid activation functions in this case; 
# if the output has more than 2 categories, then use the softmax function - another version of sigmoid function

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# if the output has more than 2 categories, then the loss = 'categorical_crossentropy'

# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 32, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)