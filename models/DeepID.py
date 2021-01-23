'''
Date: 2021-01-09 20:24:04
LastEditors: Rustle Karl
LastEditTime: 2021-01-09 22:10:15
'''

from tensorflow.keras.layers import (Activation, Add, Conv2D, Dense, Dropout,
                                     Flatten, Input, MaxPooling2D)
from tensorflow.keras.models import Model


def DeepIDModel():
    _input = Input(shape=(55, 47, 3))

    x = Conv2D(20, (4, 4), name='Conv1', activation='relu', input_shape=(55, 47, 3))(_input)
    x = MaxPooling2D(pool_size=2, strides=2, name='Pool1')(x)
    x = Dropout(rate=0.99, name='D1')(x)

    x = Conv2D(40, (3, 3), name='Conv2', activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2, name='Pool2')(x)
    x = Dropout(rate=0.99, name='D2')(x)

    x = Conv2D(60, (3, 3), name='Conv3', activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2, name='Pool3')(x)
    x = Dropout(rate=0.99, name='D3')(x)

    x1 = Flatten()(x)
    fc11 = Dense(160, name='fc11')(x1)

    x2 = Conv2D(80, (2, 2), name='Conv4', activation='relu')(x)
    x2 = Flatten()(x2)
    fc12 = Dense(160, name='fc12')(x2)

    y = Add()([fc11, fc12])
    y = Activation('relu', name='deepid')(y)

    return Model(inputs=[_input], outputs=y)