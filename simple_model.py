import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics

def create_model(input_size):
    num_labels = 2
    filter_size = 2

    # Construct model
    model = Sequential()

    model.add(Dense(256, input_shape=(input_size,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    return model

def load_trained_model(weights_path, input_size):
    model = create_model(input_size)
    model.load_weights(weights_path)
    return model
