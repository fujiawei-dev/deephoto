'''
Date: 2021-01-15 08:45:14
LastEditors: Rustle Karl
LastEditTime: 2021-01-20 07:36:57
'''

from logger import log

from client.api import delete_photo
from client.orm import Photoprism
from configs import is_file


def clean_photos():
    '''清理不存在的文件'''
    photoprism = Photoprism()
    photos = photoprism.select_deleted_photos()

    for photo in photos:
        file, photo_path = is_file(photo.files.file_name)

        if not file:
            log.warning("%s is not exist" % photo_path)
            delete_photo(photo.photo_uid)

    log.info("clean photos finished")


if __name__ == '__main__':
    clean_photos()
