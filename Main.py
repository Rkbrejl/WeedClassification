from keras.models import Sequential, Model
from keras.applications import ResNet50
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import random

base_model = ResNet50(weights='imagenet', include_top=False)
for layer in base_model.layers:
    layer.trainable = False

fc = base_model.output
fc = Dense(200,activation='relu')(fc)

model = Model(input=base_model.input,output=fc)

model.compile(loss='mse',optimizer=Adam())
