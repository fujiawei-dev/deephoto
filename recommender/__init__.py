'''
Date: 2021-01-14 17:04:28
LastEditors: Rustle Karl
LastEditTime: 2021-01-15 16:25:31
'''
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

from recommender.model import load_model


# TODO 时间惩罚因子，最后浏览时间
def get_rank(like: int, views: int, last_visited_at=None) -> int:
    if views == 0:
        rank = 0
    elif like < 0:
        rank = np.log2(views) / np.log2(16)
    elif like > 0:
        rank = np.log2(views)
    else:
        rank = np.log2(views) / np.log2(4)

    return int(np.floor(rank)) if rank < 9 else 9


def get_distance(prediction: np.ndarray) -> float:
    return sum(prediction * np.array(range(0, 10)))


def load_image(image_path, rank):
    image = tf.io.read_file(image_path)  # 读取图片
    image = tf.image.decode_image(image, channels=3, expand_animations=False)  # gif is 4D
    image = tf.image.resize(image, [300, 300])
    return image, rank


class Recommender(object):

    def __init__(self):
        self.model = load_model()
        self.checkpoint_path = "training/recommender.ckpt"
        self.checkpoint_dir = "training"

        if os.path.isfile(self.checkpoint_path):
            self.model.load_weights(self.checkpoint_path)

        # 创建一个保存模型权重的回调
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.checkpoint_path, save_weights_only=True, verbose=0)

    def fit(self, images_list: list, ranks_list: list):
        ds = tf.data.Dataset.from_tensor_slices((images_list, ranks_list)).map(load_image)
        return self.model.fit(ds.batch(32), callbacks=[self.cp_callback])  # 通过回调训练

    def predict(self, image_path):
        img = keras.preprocessing.image.load_img(image_path, target_size=(300, 300))
        array = keras.preprocessing.image.img_to_array(img)
        return self.model.predict(tf.expand_dims(array, 0))[0]

    def get_distance(self, image_path):
        score = tf.nn.softmax(self.predict(image_path))
        return get_distance(score)
