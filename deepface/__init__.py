'''
Date: 2021-01-12 17:31:05
LastEditors: Rustle Karl
LastEditTime: 2021-01-12 22:56:52
'''
import os
from typing import List, Union

from deepface.detector import FaceDetector
from deepface.distance import Distance, findCosineDistance, findDistance, findEuclideanDistance, findInputShape, \
    findNearest, findThreshold, l2Normalize
from models import Models, load

from logger import get_logger

log = get_logger("deepface", level="warning")


def verify(image_pairs: List[Union[list, tuple]], verify_backend=Models.VGGFace,
           detect_backend=Models.MTCNN, metric=Distance.Cosine):
    model = load(verify_backend)
    face = FaceDetector(detect_backend)

    results = []
    threshold = findThreshold(verify_backend, metric)
    target_size = findInputShape(model)[:2][::-1]
    for pair in image_pairs:
        if len(pair) != 2:
            raise ValueError("Invalid image_pair passed to verify function: ", image_pairs)

        image_x, image_y = pair
        img_x = face.preprocess(image_x, target_size=target_size)
        img_y = face.preprocess(image_y, target_size=target_size)

        # find embeddings
        img_x_embed = model.predict(img_x)[0, :]
        img_y_embed = model.predict(img_y)[0, :]

        distance = findDistance(img_x_embed, img_y_embed, metric)
        if distance <= threshold:
            identified = True
        else:
            identified = False

        results.append({
            "image_x": image_x,
            "image_y": image_y,
            "verified": identified,
            "distance": distance,
            "threshold": threshold,
        })

    return results


def find(image_path, image_dir, verify_backend=Models.VGGFace,
         detect_backend=Models.MTCNN, metric=Distance.Cosine) -> list:
    if not (os.path.isdir(image_dir) and os.path.isfile(image_path)):
        raise FileNotFoundError

    model = load(verify_backend)
    face = FaceDetector(detect_backend)

    target_size = findInputShape(model)[:2][::-1]
    img_x = face.preprocess(image_path, target_size=target_size)
    img_x_embed = model.predict(img_x)[0, :]
    threshold = findThreshold(verify_backend, metric)

    results = []
    for image in os.listdir(image_dir):
        image_y = os.path.join(image_dir, image)
        img_y = face.preprocess(image_y, target_size=target_size)
        img_y_embed = model.predict(img_y)[0, :]

        if findDistance(img_x_embed, img_y_embed, metric) <= threshold:
            results.append(image_y)

    return results


class DeepFace(object):
    target = None  # 基准图片
    target_embed = None

    def __init__(self, verify_backend=Models.VGGFace,
                 detect_backend=Models.MTCNN, metric=Distance.Cosine):
        self.model = load(verify_backend)
        self.face = FaceDetector(detect_backend)
        self.metric = metric
        self.target_size = findInputShape(self.model)[:2][::-1]
        self.threshold = findThreshold(verify_backend, metric)

    def preprocess_predict(self, image):
        if (img := self.face.preprocess(image, target_size=self.target_size)) is None:
            return None
        return self.model.predict(img)[0, :]

    def set_target(self, image):
        self.target, self.target_embed = image, self.preprocess_predict(image)

    def set_target_embed(self, target_embed):
        self.target_embed = target_embed

    def detect(self, output=None) -> bool:
        detected, _ = self.face.detect(self.target, output)
        return detected

    def __verify(self):
        if self.target_embed is None:
            raise ValueError('Please set a target image or embed first! Must be a portrait.')

    def find_distance(self, img_embed) -> float:
        self.__verify()
        return findDistance(self.target_embed, img_embed)

    def verify_embed(self, img_embed) -> bool:
        return self.find_distance(img_embed) <= self.threshold

    def verify_embed_list(self, img_embed_list: Union[list, tuple]) -> list:
        return list(map(self.verify_embed, img_embed_list))

    def verify(self, image):
        return self.verify_embed(self.preprocess_predict(image))

    def verify_list(self, images: Union[list, tuple]) -> list:
        return list(map(self.verify_embed, map(self.preprocess_predict, images)))

    def find_nearest(self, images: Union[list, tuple]) -> int:
        pass

    def find_embed_nearest(self, img_embed_list: Union[list, tuple]) -> (bool, int):
        if not img_embed_list:
            return False, 0
        ds = list(map(self.find_distance, img_embed_list))
        d = findNearest(ds)
        return ds[d] <= self.threshold, d
