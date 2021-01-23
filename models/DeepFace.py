'''
Date: 2021-01-12 17:31:02
LastEditors: Rustle Karl
LastEditTime: 2021-01-12 19:10:27
'''
from tensorflow.keras.layers import (Convolution2D, Dense, Dropout, Flatten,
                                     LocallyConnected2D, MaxPooling2D)
from tensorflow.keras.models import Sequential


def DeepFaceModel():
    model = Sequential()
    model.add(Convolution2D(32, (11, 11), activation='relu', name='C1', input_shape=(152, 152, 3)))
    model.add(MaxPooling2D(pool_size=3, strides=2, padding='same', name='M2'))
    model.add(Convolution2D(16, (9, 9), activation='relu', name='C3'))
    model.add(LocallyConnected2D(16, (9, 9), activation='relu', name='L4'))
    model.add(LocallyConnected2D(16, (7, 7), strides=2, activation='relu', name='L5'))
    model.add(LocallyConnected2D(16, (5, 5), activation='relu', name='L6'))
    model.add(Flatten(name='F0'))
    model.add(Dense(4096, activation='relu', name='F7'))
    model.add(Dropout(rate=0.5, name='D0'))
    model.add(Dense(8631, activation='softmax', name='F8'))

    return model
