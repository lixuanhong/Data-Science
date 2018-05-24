# Artificial Neural Network

# Importing the dataset
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[4:14]

# Encoding the categorical variables as factor - deep learning requires as.numeric
dataset$Geography = as.numeric(factor(dataset$Geography, 
                           levels = c('France', 'Spain', 'Germany'),
                           labels = c(1, 2, 3)))

dataset$Gender = as.numeric(factor(dataset$Gender, 
                                      levels = c('Female', 'Male'),
                                      labels = c(1, 2)))


# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling - required by deep learning as it involves a lot of computation
training_set[-11] = scale(training_set[-11])
test_set[-11] = scale(test_set[-11])

# Fitting ANN to the Training set
# install.packages('h2o')
library(h2o)
h2o.init(nthread = -1) #connect to the H20 cluster/server for computation (much faster for CPU)
classifier = h2o.deeplearning(y = 'Exited',
                              training_frame = as.h2o(training_set), #as.h2o() convert the dataframe to h2o object
                              activation = 'Rectifier',
                              hidden = c(6, 6), # c() is the function to define a vector; the first number is to define the number nodes in the first hidden layer, the second number is to define the number of nodes in the second hidden layer
                              epochs = 100,
                              train_samples_per_iteration = -2) # this is the batch number/ -2 represent auto-tuning

# Predicting the Test set results
prob_pred = h2o.predict(classifier, newdata = as.h2o(test_set[-11]))
y_pred = (prob_pred > 0.5) #return a boolean
y_pred = as.vector(y_pred) #conver the h2o object to a vector

# Making the Confusion Matrix
cm = table(test_set[, 11], y_pred)

h2o.shutdown() #disconnect from the server