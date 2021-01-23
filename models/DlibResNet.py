'''
Date: 2021-01-09 23:33:30
LastEditors: Rustle Karl
LastEditTime: 2021-01-10 08:51:45
'''

import dlib  # 19.20.0
import numpy as np


class DlibResNet(object):

    def __init__(self, weights):
        self.layers = [DlibMetaData()]
        model = dlib.face_recognition_model_v1(weights)
        self.__model = model

    def predict(self, img_aligned):

        # functions.detectFace returns 4 dimensional images
        if len(img_aligned.shape) == 4:
            img_aligned = img_aligned[0]

        # functions.detectFace returns bgr images
        img_aligned = img_aligned[:, :, ::-1]  # bgr to rgb

        # deepface.detectFace returns an array in scale of [0, 1] but dlib expects in scale of [0, 255]
        if img_aligned.max() <= 1:
            img_aligned = img_aligned * 255

        img_aligned = img_aligned.astype(np.uint8)

        model = self.__model

        img_representation = model.compute_face_descriptor(img_aligned)
        img_representation = np.array(img_representation)
        img_representation = np.expand_dims(img_representation, axis=0)

        return img_representation


class DlibMetaData:
    def __init__(self):
        self.input_shape = [[1, 150, 150, 3]]
