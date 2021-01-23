'''
Date: 2021-01-12 18:04:58
LastEditors: Rustle Karl
LastEditTime: 2021-01-12 19:09:57
'''
import cv2


def load_caffe(prototxt, caffe):
    return cv2.dnn.readNetFromCaffe(prototxt, caffe)


def load_haarcascade(prototxt):
    return cv2.CascadeClassifier(prototxt)


def OpenCVDetector(prototxt, caffe=None):
    if caffe:
        return load_caffe(prototxt, caffe)
    return load_haarcascade(prototxt)
