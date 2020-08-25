import tensorflow as tf
import numpy as np
import keras
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.models import model_from_json

def model1():
    X_input = Input((100,100,1))
    X = ZeroPadding2D((1,1))(X_input)

    X = Conv2D(64, (5,5), strides = (1,1), name = 'conv0')(X)
    X = Conv2D(64, (5,5), strides = (1,1),padding='valid', name = 'conv1')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool1')(X)

    X = Conv2D(128, (5,5), strides = (1,1),padding ='same',name = 'conv2')(X)
    X = Conv2D(128, (5,5), strides = (1,1),padding='valid', name = 'conv3')(X)
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool2')(X)

    X = Conv2D(256, (7,7), strides = (1,1),padding='same', name = 'conv4')(X)
    X = Conv2D(256, (7,7), strides = (1,1),padding='valid', name = 'conv5')(X)
    X = BatchNormalization(axis = 3, name = 'bn2')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool3')(X)

    X = Conv2D(512, (7,7), strides = (1,1),padding='same', name = 'conv6')(X)
    X = BatchNormalization(axis = 3, name = 'bn3')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool4')(X)

    X = Flatten()(X)
    X=Dense(4096)(X)
    X=Dense(2048)(X)
    X=Dense(1024)(X)
    X=Dense(512)(X)
    X=Dense(256)(X)
    X=Dense(128)(X)
    X=Dense(64)(X)
    X=Dense(32)(X)
    X=Dense(16)(X)
    X=Dense(8)(X)
    X=Dense(4)(X)
    X=Dense(2)(X)
    X=Dense(3,activation ='softmax')(X)

    model = Model(inputs = X_input, outputs = X, name='xraymodel')
    model.summary()
    return model
