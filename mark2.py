#!/usr/bin/env python3
import numpy as np
import os
import csv
import model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import keras
from skimage.io import imread
from skimage.transform import resize
import cv2





File = open("./input.csv")
filedata = list(csv.reader(File, delimiter=";"))
X_train_filenames=[]
y_train = []
for i in filedata:
    X_train_filenames.append(i[0])
    y_train.append(int(i[1]))
y_train = np.array(y_train)

X = np.zeros((len(X_train_filenames),100,100,1))
for i in range(len( X_train_filenames)):
    img = cv2.imread("./chest_xray/allimages/"+X_train_filenames[i],0)
    img = resize(img,(100,100,1))
    X[i,:,:,:]=img
    print(i)
np.save("train.npy",X)



X_train= np.load("train.npy")
print(X_train.shape)
#X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
mod1 = model.model1()
opt = keras.optimizers.Adam()
mod1.compile(optimizer=opt,loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=["accuracy"])
mod1.fit(X_train,y_train, epochs = 30, batch_size=100)
model_json = mod1.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
mod1.save_weights("model.h5")
print("Saved model to disk")
