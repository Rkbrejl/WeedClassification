from keras.models import Sequential, Model
from keras.applications import ResNet50
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.datasets import cifar100
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import random

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

def prepare_input_data(X_train, X_test):
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    return X_train, X_test

def prepare_output_data(y_train, y_test):
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    return y_train, y_test

x_train, x_test = prepare_input_data(x_train, x_test)
y_train, y_test = prepare_output_data(y_train, y_test)

base_model = ResNet50(input_shape=(None, None, 3), weights='imagenet', include_top=False)
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
fc = Dense(256, activation='relu')(x)
sm = Dense(100, activation='softmax')(fc)

model = Model(input=base_model.input, output=sm)

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['acc'])

model.fit(x_train, y_train, batch_size=32, nb_epoch=20, verbose=1)
