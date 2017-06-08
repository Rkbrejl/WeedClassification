import numpy as np
from keras.utils import np_utils
from sklearn import preprocessing
import skimage.transform
import skimage.color
from tqdm import trange
import os.path
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.applications import ResNet50, VGG16
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import CSVLogger
from keras.losses import categorical_crossentropy
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import random

file = open('natural_train_species.txt', 'r')

y_data = []
ids = []
plant = True
for line in file:
    if plant:
        y_data.append(line.split())
    else:
        ids.append(line.split())
    plant = not plant
file.close()

ids = np.array(ids).astype('int32')
# ids = ids[1:50]

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(y_data)

y_encoded = label_encoder.transform(y_data)

y_train = np_utils.to_categorical(y_encoded)

path = 'Plants/train/'

RESOLUTION = (200, 200, 3)
x_train = np.zeros([len(ids)] + list(RESOLUTION))

show = False

def preprocess(img):
    img = skimage.transform.resize(img, RESOLUTION)
    img = img.astype('float32')
    return img

print('Processing images\n ---------------- \n\n')
for i in trange(len(ids), leave=False):
    this_path = path+str(ids[i, 0])+'.jpg'
    if os.path.isfile(this_path):
        some_img = Image.open(this_path)
        this_img = np.asarray(some_img.convert())
        processed_img = preprocess(this_img)
        if show:
            plt.imshow(this_img)
            plt.show()
            plt.imshow(processed_img)
            plt.show()
        x_train[i, :, :, :] = processed_img

print('Images processed...\n\n')

y_train = y_train[0:len(ids)]

print('Getting weights for model...\n\n')

base_model = ResNet50(input_shape=(None, None, 3), weights='imagenet', include_top=False)
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
fc = Dense(256, activation='relu')(x)
sm = Dense(len(y_train[1]), activation='softmax')(fc)

model = Model(input=base_model.input, output=sm)

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['acc'])

csv_logger = CSVLogger('trainlog.csv')
print('Starting training...\n\n')
model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=1, validation_split=0.2, callbacks=[csv_logger])