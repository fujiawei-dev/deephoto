'''
Date: 2021-01-12 18:04:07
LastEditors: Rustle Karl
LastEditTime: 2021-01-12 22:30:49
'''
import base64
import os
from typing import Tuple, Union

import cv2
import dlib
import numpy as np
import pandas as pd
from logger import get_logger
from PIL import Image
from tensorflow.keras import preprocessing

from configs import Path
from deepface import distance
from models import load, Models

log = get_logger("detector", level="info")


def read_image(image: Union[str, np.ndarray]) -> np.ndarray:
    if image is None:
        return None
    elif isinstance(image, np.ndarray):
        img = image
    elif isinstance(image, str) and image[0:11] == 'data:image/':
        enc = image.split(',', 1)[1]
        array = np.fromstring(base64.b64decode(enc), np.uint8)
        img = cv2.imdecode(array, cv2.IMREAD_COLOR)
    elif os.path.isfile(image):
        img = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_COLOR)
    elif os.path.splitext(image)[-1] not in (".jpg", ".jpeg", ".png"):
        return None
    else:
        raise NotImplementedError(image)

    if img is None:
        log.error(image)
        return None

    h, w = img.shape[:2]
    hn, wn = 720, 1280
    if h / w >= hn / wn:
        img = cv2.resize(img, (int(w * hn / h), hn))
    else:
        img = cv2.resize(img, (wn, int(h * wn / w)))

    return img


def alignment_procedure(img: np.ndarray, left_eye: Union[tuple, list], right_eye: Union[tuple, list]) -> np.ndarray:
    # this function aligns given face in img based on left and right eye coordinates

    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    # find rotation direction
    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # rotate inverse direction of clock

    # find length of triangle edges
    a = distance.findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
    b = distance.findEuclideanDistance(
            np.array(right_eye), np.array(point_3rd))
    c = distance.findEuclideanDistance(np.array(right_eye), np.array(left_eye))

    # apply cosine rule
    if b != 0 and c != 0:  # this multiplication causes division by zero in cos_a calculation
        cos_a = (b * b + c * c - a * a) / (2 * b * c)
        angle = np.arccos(cos_a)  # angle in radian
        angle = (angle * 180) / np.pi  # radian to degree

        # rotate base image
        if direction == -1:
            angle = 90 - angle

        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))

    return img  # return img anyway


class FaceDetector(object):
    __eye_detector = None
    __shape_predictor = None

    def __init__(self, backend=Models.OpenCVSSD):
        self.__detector = load(backend)
        if backend == Models.OpenCV or backend == Models.OpenCVSSD:
            self.__eye_detector = load(Models.OpenCVEye)
        elif backend == Models.DlibFace:
            self.__shape_predictor = load(Models.DlibShape)
        self.__backend = backend

    def align(self, image: Union[str, np.ndarray] = None, output: str = None) -> Union[bool, np.ndarray]:
        img = read_image(image)
        out = None

        if self.__backend == Models.OpenCV or self.__backend == Models.OpenCVSSD:  # 极差
            # eye detector expects gray scale image
            detected_face_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            eyes = self.__eye_detector.detectMultiScale(detected_face_gray)

            if len(eyes) >= 2:

                # find the largest 2 eye
                base_eyes = eyes[:, 2]
                items = []
                for i in range(0, len(base_eyes)):
                    item = (base_eyes[i], i)
                    items.append(item)
                df = pd.DataFrame(items, columns=['length', 'idx']). \
                    sort_values(by=['length'], ascending=False)

                # eyes variable stores the largest 2 eye
                eyes = eyes[df.idx.values[0:2]]

                # decide left and right eye
                eye_1 = eyes[0]
                eye_2 = eyes[1]
                if eye_1[0] < eye_2[0]:
                    left_eye = eye_1
                    right_eye = eye_2
                else:
                    left_eye = eye_2
                    right_eye = eye_1

                # find center of eyes
                left_eye = (int(left_eye[0] + (left_eye[2] / 2)),
                            int(left_eye[1] + (left_eye[3] / 2)))
                right_eye = (int(right_eye[0] + (right_eye[2] / 2)),
                             int(right_eye[1] + (right_eye[3] / 2)))
                img = alignment_procedure(img, left_eye, right_eye)

            out = img

        if self.__backend == Models.DlibFace:
            detections = self.__detector(img, 1)
            if len(detections) > 0:
                detected_face = detections[0]
                img_shape = self.__shape_predictor(img, detected_face)
                img = dlib.get_face_chip(img, img_shape, size=img.shape[0])
            out = img  # return img anyway

        if self.__backend == Models.MTCNN:
            # mtcnn expects RGB but OpenCV read BGR
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            detections = self.__detector.detect_faces(img_rgb)

            if len(detections) > 0:
                detection = detections[0]

                keypoints = detection['keypoints']
                left_eye = keypoints['left_eye']
                right_eye = keypoints['right_eye']

                img = alignment_procedure(img, left_eye, right_eye)

            out = img  # return img anyway

        if out is not None:
            if output:
                cv2.imwrite(output, out)
            return out

        detectors = [Models.OpenCV, Models.OpenCVSSD, Models.DlibFace, Models.MTCNN]
        raise ValueError('Valid backends are ', detectors, ' but you passed ', self.__backend)

    def detect(self, image: Union[str, np.ndarray], output: str = None) -> Tuple[bool, np.ndarray]:
        if (img := read_image(image)) is None:
            return False, None

        out = None

        if self.__backend == Models.MTCNN:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # mtcnn expects RGB but OpenCV read BGR
            detections = self.__detector.detect_faces(img_rgb)

            if len(detections) == 0:
                return False, None

            detection = detections[0]
            x, y, w, h = detection['box']
            confidence = detection['confidence']

            if h + w < 100 or confidence < 0.95:
                log.info("Discard: %s" % {"confidence": confidence, "height": h, "width": w})
                return False, None

            out = img[int(y):int(y + h), int(x):int(x + w)]

        elif self.__backend == Models.OpenCV:  # 最差
            detections = self.__detector.detectMultiScale(img, 1.3, 5)
            if len(detections) == 0:
                return False, None
            x, y, w, h = detections[0]  # focus on the 1st face found in the image
            out = img[int(y):int(y + h), int(x):int(x + w)]

        elif self.__backend == Models.OpenCVSSD:  # 最快
            raw_img = img.copy()  # we will restore raw_img to img later

            ssd_labels = ['img_id', 'is_face', 'confidence', 'left', 'top', 'right', 'bottom']
            target_size = (300, 300)
            original_size = img.shape
            img = cv2.resize(img, target_size)

            aspect_ratio_x = (original_size[1] / target_size[1])
            aspect_ratio_y = (original_size[0] / target_size[0])

            image_blob = cv2.dnn.blobFromImage(image=img)

            self.__detector.setInput(image_blob)
            detections = self.__detector.forward()

            detections_df = pd.DataFrame(detections[0][0], columns=ssd_labels)

            # 0: background, 1: face
            detections_df = detections_df[detections_df['is_face'] == 1]
            detections_df = detections_df[detections_df['confidence'] >= 0.90]

            detections_df['left'] = (detections_df['left'] * 300).astype(int)
            detections_df['bottom'] = (detections_df['bottom'] * 300).astype(int)
            detections_df['right'] = (detections_df['right'] * 300).astype(int)
            detections_df['top'] = (detections_df['top'] * 300).astype(int)

            if detections_df.shape[0] == 0:
                return False, None

            instance = detections_df.iloc[0]  # get the first face in the image
            left = instance['left'] * aspect_ratio_x
            right = instance['right'] * aspect_ratio_x
            bottom = instance['bottom'] * aspect_ratio_y
            top = instance['top'] * aspect_ratio_y

            out = raw_img[int(top):int(bottom), int(left):int(right)]

        elif self.__backend == Models.DlibFace:  # slowest
            detections = self.__detector(img, 1)

            if len(detections) == 0:
                return False, None

            detection = detections[0]
            left = detection.left()
            right = detection.right()
            top = detection.top()
            bottom = detection.bottom()
            out = img[top:bottom, left:right]

        if out is not None:
            if output:
                try:
                    cv2.imwrite(output, out)
                except cv2.error:
                    pass
            return True, out

        detectors = [Models.OpenCV, Models.OpenCVSSD, Models.DlibFace, Models.MTCNN]
        raise ValueError('Valid backends are ', detectors, ' but you passed ', self.__backend)

    def preprocess(self, image: Union[str, np.ndarray], target_size=(224, 224),
                   grayscale=False, output=None) -> Union[bool, np.ndarray]:
        log.debug("Detecting: %s" % Path(image).name)

        img = read_image(image)
        if img is None:
            return None
        raw_img = img.copy()

        detected, img = self.detect(img)
        if not detected:
            return None

        if img.shape[0] > 0 and img.shape[1] > 0:
            img = self.align(img)
        else:
            img = raw_img.copy()

        # post-processing
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_pixels = preprocessing.image.img_to_array(cv2.resize(img, target_size))
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255  # normalize input in [0, 1]

        if output:
            preprocessing.image.array_to_img(img_pixels.squeeze(0)).save(output)

        return img_pixels
