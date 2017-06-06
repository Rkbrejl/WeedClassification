import numpy as np
from keras.utils import np_utils
from sklearn import preprocessing
import skimage.transform
import skimage.color
from tqdm import trange
import os.path
from PIL import Image
import matplotlib.pyplot as plt

file = open('train_species.txt', 'r')

y_data = []
for line in file:
    y_data.append(line.split())
file.close()

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(y_data)

y_encoded = label_encoder.transform(y_data)

y_train = np_utils.to_categorical(y_encoded)

path = 'Plants/train/'
highest_id = 200
RESOLUTION = (200, 200, 3)
x_train = []

show = False

def preprocess(img):
    img = skimage.transform.resize(img, RESOLUTION)
    img = img.astype('float32')
    return img

print('Processing images\n ---------------- \n\n')
for i in trange(highest_id, leave=False):
    this_path = path+str(i)+'.jpg'
    if os.path.isfile(this_path):
        some_img = Image.open(this_path)
        this_img = np.asarray(some_img.convert())
        if show:
            plt.imshow(this_img)
            plt.show()
            plt.imshow(preprocess(this_img))
            plt.show()
        x_train.append(preprocess(this_img))

x_train = np.array(x_train)
y_train = y_train[0:len(x_train)]

from keras.models import Sequential, Model
from keras.applications import ResNet50, VGG16
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import CSVLogger
from keras.losses import categorical_crossentropy
from keras.datasets import cifar100
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import random

# (x_train, y_train), (x_test, y_test) = cifar100.load_data()

# print(np.array(x_train).shape)

def prepare_input_data(X_train, X_test):
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    return X_train, X_test

def prepare_output_data(y_train, y_test):
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    return y_train, y_test

# x_train, x_test = prepare_input_data(x_train, x_test)
# y_train, y_test = prepare_output_data(y_train, y_test)

base_model = ResNet50(input_shape=(None, None, 3), weights='imagenet', include_top=False)
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
fc = Dense(1000, activation='relu')(x)
sm = Dense(250, activation='softmax')(fc)


model = Model(input=base_model.input, output=sm)

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['acc'])

csv_logger = CSVLogger('trainlog.csv')
model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=1,validation_split=0.2, callbacks=[csv_logger])