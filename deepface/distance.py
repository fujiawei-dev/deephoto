'''
Date: 2021-01-12 17:56:09
LastEditors: Rustle Karl
LastEditTime: 2021-01-12 22:30:57
'''
import numpy as np
from tensorflow.python.keras import Model
from typing import List

from models import Models


class Distance(object):
    Cosine = 'cosine'
    Euclidean = 'euclidean'
    EuclideanL2 = 'euclidean_l2'


def findCosineDistance(x, y):
    a = np.matmul(np.transpose(x), y)
    b = np.sum(np.multiply(x, x))
    c = np.sum(np.multiply(y, y))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def findEuclideanDistance(x, y):
    dist = x - y
    dist = np.sum(np.multiply(dist, dist))
    dist = np.sqrt(dist)
    return dist


def l2Normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))


def findEuclideanL2Distance(x, y):
    return findEuclideanDistance(l2Normalize(x), l2Normalize(y))


def findDistance(x, y, metric=Distance.Cosine):
    if metric == Distance.Cosine:
        return findCosineDistance(x, y)
    elif metric == Distance.EuclideanL2:
        return findEuclideanDistance(l2Normalize(x), l2Normalize(y))
    return findEuclideanDistance(x, y)


def findNearest(ns: List[int]) -> int:
    index = 0
    last = ns[0]
    for i in range(1, len(ns)):
        if ns[i] < last:
            index = i
    return index


DefaultThreshold = {'cosine': 0.40, 'euclidean': 0.55, 'euclidean_l2': 0.75}

Thresholds = {
    # Models.VGGFace: {'cosine': 0.40, 'euclidean': 0.55, 'euclidean_l2': 0.75},
    Models.VGGFace: {'cosine': 0.35, 'euclidean': 0.55, 'euclidean_l2': 0.75},
    Models.OpenFace: {'cosine': 0.10, 'euclidean': 0.55, 'euclidean_l2': 0.55},
    Models.FaceNet: {'cosine': 0.40, 'euclidean': 10, 'euclidean_l2': 0.80},
    Models.DeepFace: {'cosine': 0.23, 'euclidean': 64, 'euclidean_l2': 0.64},
    Models.DeepID: {'cosine': 0.015, 'euclidean': 45, 'euclidean_l2': 0.17},
    Models.DlibResNet: {'cosine': 0.07, 'euclidean': 0.6, 'euclidean_l2': 0.6},
    Models.ArcFace: {'cosine': 0.6871912959056619, 'euclidean': 4.1591468986978075, 'euclidean_l2': 1.1315718048269017}
}


def findThreshold(model_type, distance_metric):
    return Thresholds.get(model_type, DefaultThreshold).get(distance_metric, 0.4)


def findInputShape(model: Model):
    input_shape = model.layers[0].input_shape

    if type(input_shape) == list:
        input_shape = input_shape[0][1:3]
    else:
        input_shape = input_shape[1:3]

    return input_shape
