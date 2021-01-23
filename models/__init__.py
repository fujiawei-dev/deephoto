'''
Date: 2021-01-12 17:31:05
LastEditors: Rustle Karl
LastEditTime: 2021-01-12 19:08:53
'''
from pathlib import Path
from typing import Union

import cv2
import dlib
import gdown
from mtcnn import MTCNN
from tensorflow.keras.models import Model

from models.ArcFace import ArcFaceModel
from models.DeepFace import DeepFaceModel
from models.DeepID import DeepIDModel
from models.DlibResNet import DlibResNet
from models.FaceNet import InceptionResNetV2
from models.OpenCV import OpenCVDetector
from models.OpenFace import OpenFaceModel
from models.VGGFace import VGGFaceModel
from models.ext.Age import AgeModel
from models.ext.Emotion import EmotionModel
from models.ext.Gender import GenderModel
from models.ext.Race import RaceModel

HOME = Path('c:/mirror/weights')
DOWNLOAD = 'http://192.168.199.208:12345/weights/'

if not HOME.exists():
    HOME.mkdir()


class Models(object):
    VGGFace = 'vggface'
    OpenFace = 'openface'
    DeepFace = 'deepface'
    FaceNet = 'facenet'
    DlibResNet = 'dlib'
    DlibFace = 'dlibface'
    DlibShape = 'dlibshape'
    DeepID = 'deepid'
    ArcFace = 'arcface'
    OpenCV = 'opencv'
    OpenCVSSD = 'opencvssd'
    OpenCVEye = 'opencveye'
    MTCNN = 'mtcnn'

    Age = 'age'
    Emotion = 'emotion'
    Gender = 'gender'
    Race = 'race'


class ModelLoader(object):
    def __init__(self, weights, model, annex=''):
        self.downloadUrl = DOWNLOAD + weights
        self.weights = HOME / weights
        self.model = model
        self.annex = HOME / annex

    def load(self) -> Union[Model, dlib.face_recognition_model_v1, dlib.shape_predictor,
                            cv2.CascadeClassifier]:
        if not self.weights.is_file():
            print("%s will be downloaded...", self.weights.name)
            gdown.download(self.downloadUrl,
                           self.weights.as_posix(),
                           quiet=False)

        if self.model == DlibResNet:
            return self.model(self.weights.as_posix())
        if self.model == dlib.shape_predictor:
            return self.model(self.weights.as_posix())
        if self.model == OpenCVDetector:
            caffe = None
            if self.annex.is_file():
                caffe = self.annex.as_posix()
            return self.model(self.weights.as_posix(), caffe)

        model = self.model()
        try:
            model.load_weights(self.weights.as_posix())
        except Exception as err:
            print("Pre-trained weight could not be loaded: %s" % err)
            print(
                    "You might try to download the pre-trained weights from the url ",
                    self.downloadUrl, " and copy it to the ", self.weights)

        return self.layer(model)

    def layer(self, model) -> Model:
        if self.model == VGGFaceModel:
            return Model(inputs=model.layers[0].input,
                         outputs=model.layers[-2].output)
        if self.model == DeepFaceModel:
            # drop F8 and D0. F7 is the representation layer.
            return Model(inputs=model.layers[0].input,
                         outputs=model.layers[-3].output)
        return model


models = {
    Models.VGGFace: ModelLoader('vgg_face_weights.h5', VGGFaceModel),
    Models.OpenFace: ModelLoader('openface_weights.h5', OpenFaceModel),
    Models.DeepFace: ModelLoader('VGGFace2_DeepFace_weights_val-0.9034.h5', DeepFaceModel),
    Models.FaceNet: ModelLoader('facenet_weights.h5', InceptionResNetV2),
    Models.DlibResNet: ModelLoader('dlib_face_recognition_resnet_model_v1.dat', DlibResNet),
    Models.DlibShape: ModelLoader('shape_predictor_5_face_landmarks.dat', dlib.shape_predictor),
    Models.DeepID: ModelLoader('deepid_keras_weights.h5', DeepIDModel),
    Models.ArcFace: ModelLoader('arcface_weights.h5', ArcFaceModel),
    Models.OpenCV: ModelLoader('haarcascade_frontalface_default.xml', OpenCVDetector),
    Models.OpenCVSSD: ModelLoader('deploy.prototxt', OpenCVDetector, 'res10_300x300_ssd_iter_140000.caffemodel'),
    Models.OpenCVEye: ModelLoader('haarcascade_eye.xml', OpenCVDetector),

    Models.Age: ModelLoader('age_model_weights.h5', AgeModel),
    Models.Emotion: ModelLoader('facial_expression_model_weights.h5', EmotionModel),
    Models.Gender: ModelLoader('gender_model_weights.h5', GenderModel),
    Models.Race: ModelLoader('race_model_single_batch.h5', RaceModel)
}


def load(backend: str = Models.VGGFace) -> Union[Model, MTCNN,
                                                 dlib.face_recognition_model_v1, dlib.shape_predictor,
                                                 cv2.CascadeClassifier]:
    if backend == Models.DlibFace:
        return dlib.get_frontal_face_detector()

    if backend == Models.MTCNN:
        return MTCNN()

    model = models.get(backend)
    if not model:
        raise NotImplementedError
    return model.load()
