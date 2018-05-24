# Convolutional Neural Network

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu')) # 32 feature detectors with 3*3
#input_shape 3 represent 3 channels for color picture 64 represents pixels
#the order of input_shape parameter is reversed in tensorflow backend compared to Theano backend input_shape=(3, 64, 64)

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2))) #usually 2*2 pooling/reduce the size of feature map by 2

# Adding a second convolutional layer to improve its accuracy
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(units = 128, activation = 'relu')) #hidden layer
classifier.add(Dense(units = 1, activation = 'sigmoid')) #output layer

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#outcome is 2 types therefore 'binary_crossentropy'

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True) #preprocess images

test_datagen = ImageDataGenerator(rescale=1./255) #preprocess images

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                    steps_per_epoch=8000,
                    epochs=10,
                    validation_data=test_set,
                    validation_steps=2000)