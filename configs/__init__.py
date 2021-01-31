'''
Date: 2021-01-16 10:32:39
LastEditors: Rustle Karl
LastEditTime: 2021-01-19 00:22:01
'''
import os
from pathlib import Path
from typing import Tuple

import tensorflow as tf

# 根路径
Originals = [
    Path("D:/OneDrive/相册"),
    # Path("H:/Cache"),
    # Path("H:/Temp")
]

Root = Path("H:/Data/photoprism/storage")
Face = Root / "face"

# if not Face.exists():
#     Face.mkdir()

# GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

try:
    gpu = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu[0], True)
except Exception:
    pass


def is_file(filepath: str) -> Tuple[bool, Path]:
    for original in Originals:
        path = original / filepath

        while len(path.suffixes) > 1:
            path = path.with_suffix('')

        if path.is_file():
            return True, path

    return False, filepath
