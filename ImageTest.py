import skimage.transform
import skimage.color
from tqdm import trange
import os.path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from statistics import mode

path = 'Plants/train/'

highest_id = 1
RESOLUTION = (800, 800, 3) # Mode resolution of the images
x_data = []

def preprocess(img):
    img = skimage.transform.resize(img, RESOLUTION)
    img = img.astype('float32')
    return img

for i in trange(highest_id, leave=False):
    this_path = path+str(i)+'.jpg'
    if os.path.isfile(this_path):
        some_img = Image.open(this_path)
        this_img = np.asarray(some_img.convert())
        plt.imshow(this_img)
        plt.show()
        plt.imshow(preprocess(this_img))
        plt.show()
        x_data.append(preprocess(this_img))

print(np.array(x_data).shape)


