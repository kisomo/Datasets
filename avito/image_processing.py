
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("/home/terrence/CODING/Python/MODELS/AvitoData"))

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications import xception
from keras.applications import inception_v3

print(os.listdir("/home/terrence/CODING/Python/MODELS/keras-pretrained-models/"))

from os import listdir, makedirs
from os.path import join, exists, expanduser
from keras.applications.vgg16 import preprocess_input

from zipfile import ZipFile
import cv2
#from dask import bag, threaded
#from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

print("----------------- VGG16 summary -----------")

model = VGG16(weights='imagenet', include_top=False)
model.summary()

#image_path = "/home/terrence/Desktop/PHONE/Camera/"

def image_data(img):
    try:        
        img = image_path + str(img) + ".jpg"
        img = image.load_img(img, target_size=(224, 224))

        x = image.img_to_array(img)  # 3 dims(3, 224, 224)
        x = np.expand_dims(x, axis=0)  # 4 dims(1, 3, 224, 224)
        x = preprocess_input(x)
        
        features = model.predict(x)
        
        feat = features.reshape((25088,))
        
        c = 784
        n = 32
        k =10

        feat2 = feat.reshape(n,c)
        pca = PCA(n_components=k)
        res = pca.fit_transform(feat2)
        res2 = res.reshape((-1,n*k))
        #return res2
        v = res2
    except:
        #return 0
        v = 0
    return v


#image_path = "/home/terrence/Desktop/PHONE/Camera/"
#x = '20140218_170332'
#print(image_data(x))

image_path = "/home/terrence/CODING/Python/MODELS/AvitoData/data/competition_files/train_jpg/"
y = '856e74b8c46edcf0c0e23444eab019bfda63687bb70a3481955cc6ab86e39df2'
print(image_data(y))
'''
img_path = '/home/terrence/Desktop/PHONE/Camera/20140218_170332.jpg'

#training data 

train = pd.read_csv("/home/terrence/CODING/Python/MODELS/AvitoData/train.csv").sample(2500)
print(train.shape)
#print(train.head(2))
#print(train.dtypes)

image_path = '/home/terrence/CODING/Python/MODELS/AvitoData/data/competition_files/train_jpg/'

#images_train = train[["image"]].drop_duplicates().dropna()

train["Image_score"] = train["image"].apply(lambda x: image_data(x))
print(train.shape)
print(train.head(2))
print(train.dtypes)
print(np.unique(train["Image_score"]))

#testing data 
test = pd.read_csv("/home/terrence/CODING/Python/MODELS/AvitoData/test.csv").sample(2500)
print(test.shape)
#print(test.head(2))
#print(test.dtypes)

image_path = '/home/terrence/CODING/Python/MODELS/AvitoData/data/competition_files/test_jpg/'

#images_test = test[["image"]].drop_duplicates().dropna()

test["Image_score"] = test["image"].apply(lambda x: image_data(x))
print(test.shape)
print(test.head(2))
print(test.dtypes)
print(np.unique(test["Image_score"]))
'''

#print(os.listdir("/home/terrence/Desktop/PHONE/Camera/"))

