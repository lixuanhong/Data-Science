# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) # quoting 3 - means ignore double quotes

# Cleaning the texts
import re
import nltk #library for natural language processing
nltk.download('stopwords') #download the tool - stopwords list
from nltk.corpus import stopwords #import the stopwords
from nltk.stem.porter import PorterStemmer #import the stem tool - to keep the root of each word, get rid of its tense
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) #only keep letters in corpuss
    review = review.lower() #convert to lower case
    review = review.split() #convert string to list
    ps = PorterStemmer() #creat a PorterStemmer object
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] #exlude word in the list of stopwords //set function make the algorithm much faster
    review = ' '.join(review) #join different words separated by space - convert list to string
    corpus.append(review)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) # keep 1500 most frequent words
X = cv.fit_transform(corpus).toarray() #create the sparse matrix .array() - to make a matrix
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = (55+91)/200 #0.73
precision = 91/(91+42) #0.684
recall = 91/(91+12) #0.883
f1_score = 2 * precision * recall / (precision + recall) #0.77

# Fitting decision tree to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = (74+68)/200 #0.71
precision = 68/(68+23) #0.75
recall = 68/(68+35) #0.66
f1_score = 2 * precision * recall / (precision + recall) #0.70

# Fitting Random Forest classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = (88+56)/200 #0.72
precision = 56/(56+9) #0.85
recall = 56/(56+47) #0.55
f1_score = 2 * precision * recall / (precision + recall) #0.667


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = (76+66)/200 #0.71
precision = 66/(66+21) #0.758
recall = 66/(66+37) #0.64
f1_score = 2 * precision * recall / (precision + recall) #0.695

# Fitting KNN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2) # use eclidean_distance
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = (74+48)/200 #0.61
precision = 48/(48+23) #0.68
recall = 48/(48+55) #0.47
f1_score = 2 * precision * recall / (precision + recall) #0.55

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0) # 'rbf', 'poly', 'sigmoid'
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = (70+74)/200 #0.72
precision = 70/(70+23) #0.75
recall = 70/(70+33) #0.68
f1_score = 2 * precision * recall / (precision + recall) #0.71


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting kernel-SVM/rbf to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = (83+64)/200 #0.735
precision = 83/(83+33) #0.716
recall = 83/(83+20) #0.806
f1_score = 2 * precision * recall / (precision + recall) #0.758

# Fitting kernel-SVM/sigmoid to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'sigmoid', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = (79+80)/200 #0.795
precision = 79/(79+17) #0.823
recall = 79/(79+24) #0.767
f1_score = 2 * precision * recall / (precision + recall) #0.794