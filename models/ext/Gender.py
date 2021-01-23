'''
Date: 2021-01-09 20:24:04
LastEditors: Rustle Karl
LastEditTime: 2021-01-09 23:29:42
'''

from tensorflow.keras.layers import Activation, Convolution2D, Flatten
from tensorflow.keras.models import Model

from models.VGGFace import VGGFaceModel


def GenderModel():
    model = VGGFaceModel()

    classes = 2
    outputs = Convolution2D(classes, (1, 1), name='predictions')(model.layers[-4].output)
    outputs = Flatten()(outputs)
    outputs = Activation('softmax')(outputs)

    return Model(inputs=model.input, outputs=outputs)
