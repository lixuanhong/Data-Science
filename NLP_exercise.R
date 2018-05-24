# Natural Language Processing

# Importing the dataset
dataset_original = read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE) #usually ignore quote in NLP

# Cleaning the texts
#install.packages('tm') - this package is used for NLP
#install.packages('SnowballC') - this package contains stopwords function
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_original$Review)) #create the corpus matrix
corpus = tm_map(corpus, content_transformer(tolower)) #convert to lower case - as.character(corpus[[1]])
corpus = tm_map(corpus, removeNumbers) #remove all numbers
corpus = tm_map(corpus, removePunctuation) #remove punctuation
corpus = tm_map(corpus, removeWords, stopwords()) #remove unrelevant words
corpus = tm_map(corpus, stemDocument) #keep the root of a word/stemming
corpus = tm_map(corpus, stripWhitespace) #remove extra space

# Creating the Bag of Words model
dtm = DocumentTermMatrix(corpus) #create the sparse matrix
dtm = removeSparseTerms(dtm, 0.999) #keep 99.9% of the most frequent words in dtm (reduce almost 1000 words)
dataset = as.data.frame(as.matrix(dtm)) #transform the sparse matrix into a dataframe
dataset$Liked = dataset_original$Liked #add the dependency column to the dataset

# Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Random Forest Classification to the Training set
library(randomForest)
classifier = randomForest(x = training_set[-692],
                          y = training_set$Liked,
                          ntree = 10)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)
accuracy = (82 + 77) / 200 #0.795
precision = 77 / (77 + 18) #0.811
recall = 77 / (77 + 23) #0.77
f1_score = 2 * precision * recall / (precision + recall) #0.79

# Fitting Logistic Regression to the Training set
classifier = glm(formula = Liked ~ .,
                 family = binomial,
                 data = training_set)

# Predicting the Test set results
prob_pred = predict(classifier, type = 'response', newdata = test_set[-692]) #remove the last column
y_pred = ifelse(prob_pred > 0.5, 1, 0)

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)
accuracy = (49 + 61) / 200 #0.55
precision = 61 / (61 + 51) #0.545
recall = 61 / (61 + 39) #0.61
f1_score = 2 * precision * recall / (precision + recall) #0.575

# Fitting K-NN to the Training set and Predicting the Test set results
library(class)
y_pred = knn(train = training_set[, -692], 
             test = test_set[, -692],
             cl = training_set[, 692],
             k = 5)

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)
accuracy = (75 + 54) / 200 #0.645
precision = 54 / (54 + 25) #0.684
recall = 54 / (54 + 46) #0.54
f1_score = 2 * precision * recall / (precision + recall) #0.603

# Fitting SVM to the Training set
library(e1071)
classifier = svm(formula = Liked ~ .,
                 data = training_set,
                 type = 'C-classification', # SVM could be used for classification and regression
                 kernel = 'linear') # 'linear' is the most basic SVM method


# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)
accuracy = (78 + 81) / 200 #0.795
precision = 81 / (81 + 22) #0.786
recall = 81 / (81 + 19) #0.81
f1_score = 2 * precision * recall / (precision + recall) #0.798


# Fitting kernel SVM to the Training set
library(e1071)
classifier = svm(formula = Liked ~ .,
                 data = training_set,
                 type = 'C-classification', # SVM could be used for classification and regression
                 kernel = 'polynomial',
                 degree = 12) 


# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)
accuracy = (71 + 82) / 200 #0.765
precision = 82 / (82 + 29) #0.739
recall = 82 / (82 + 18) #0.82
f1_score = 2 * precision * recall / (precision + recall) #0.777

# Fitting Decision Tree Classification to the Training set
library(rpart)
classifier = rpart(formula = Liked ~ .,
                   data = training_set)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692], type = 'class')

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)
accuracy = (85 + 57) / 200 #0.71
precision = 57 / (57 + 15) #0.792
recall = 57 / (57 + 43) #0.57
f1_score = 2 * precision * recall / (precision + recall) #0.663

# Fitting CART Classification to the Training set
library(rpart)
classifier = rpart(formula = Liked ~ .,
                   data = training_set,
                   method = 'class',
                   control = rpart.control(cp = 0.0, minsplit = 2))

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692], type = 'class')

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)
accuracy = (84 + 64) / 200 #0.74
precision = 64 / (64 + 16) #0.8
recall = 64 / (64 + 36) #0.64
f1_score = 2 * precision * recall / (precision + recall) #0.711

# Fitting Maximum Entropy Classification to the Training set
#install.packages('maxent')
library(maxent)
classifier = maxent(feature_matrix = training_set[-692],
                    code_vector = training_set$Liked)

# Predicting the Test set results
y_pred = predict(classifier, test_set[-692])
y_pred = y_pred[,1]

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)
accuracy = (83 + 69) / 200 #0.76
precision = 83 / (83 + 31) #0.728
recall = 83 / (83 + 17) #0.83
f1_score = 2 * precision * recall / (precision + recall) #0.776

# Fitting C5.0 Classification to the Training set
#install.packages('C50')
library(C50)
classifier = C5.0(x = training_set[-692],
                  y = training_set$Liked)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692], type = 'class')

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)
accuracy = (91 + 37) / 200 #0.64
precision = 37 / (37 + 9) #0.804
recall = 37 / (37 + 63) #0.37
f1_score = 2 * precision * recall / (precision + recall) #0.506


# Fitting Naive Bayes to the Training set
library(e1071)
classifier = naiveBayes(x = training_set[-692],
                        y = training_set$Liked)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)
accuracy = (96 + 5) / 200 #0.505
precision = 96 / (96 + 95) #0.503
recall = 96 / (96 + 4) #0.96
f1_score = 2 * precision * recall / (precision + recall) #0.659