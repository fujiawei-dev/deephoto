'''
Date: 2021-01-15 08:45:14
LastEditors: Rustle Karl
LastEditTime: 2021-01-20 07:36:57
'''
from threading import Semaphore

from logger import log
from send2trash import send2trash

from client.api import add_photos_to_album, create_album, delete_photo
from client.orm import Photoprism
from configs import Face, is_file
from deepface import DeepFace
from recommender import get_rank, Recommender

# 数据库
photoprism = Photoprism()
albums = photoprism.select_deepface_albums()

# 机器学习
deepface = DeepFace()
recommender = Recommender()

# 信号量
analysis_face_semaphore = Semaphore(1)  # TODO 分布式
recommend_photo_fit_semaphore = Semaphore(1)


def analysis_face(save_face=True):
    if analysis_face_semaphore.acquire(blocking=False):

        while photos := photoprism.select_deepface_photos():
            for photo in photos:
                file, photo_path = is_file(photo.files.file_name)

                if not file:
                    log.warning("%s is not exist" % photo_path)
                    delete_photo(photo.photo_uid)
                    continue

                id_ = photo.id
                learned = photo.learned
                deepface.set_target(photo_path.as_posix())

                if save_face:
                    save_face = (Face / (photo_path.stem + "_face.jpg")).as_posix()

                portrait_id = deepface.target_embed
                if portrait_id is None:  # not deepface.detect(output=save_face):
                    portrait = 1
                else:
                    portrait = 2
                    verify, index = deepface.find_embed_nearest(albums.values())
                    album_uid = albums.key(index)
                    if not verify:
                        if not (album_uid := create_album(portrait_id)):
                            continue
                        albums.add(album_uid, portrait_id)
                    add_photos_to_album(album_uid, photo.photo_uid)

                learned += 1
                photoprism.update_deepface_photos(id_, learned, portrait, portrait_id)

        log.info("At present, all photos have been analyzed")
        analysis_face_semaphore.release()

    log.warning("The number of threads has reached the maximum")


def recommend_photo_fit():
    if recommend_photo_fit_semaphore.acquire(blocking=False):

        while photos := photoprism.select_recommender_fit():

            delete_list = []
            image_paths = []
            ranks = []

            for photo in photos:
                file, photo_path = is_file(photo.files.file_name)

                if not file:
                    log.warning("%s is not exist" % photo_path)
                    delete_photo(photo.photo_uid)
                    continue

                views = photo.views
                last_visited_at = photo.last_visited_at
                like = photo.like
                rank = get_rank(like, views, last_visited_at)

                image_paths.append(photo_path)
                ranks.append(rank)

                if photo.deleted_at and photo.learned == 2:
                    log.warning("%s will be deleted" % photo_path)
                    delete_list.append((photo.photo_uid, photo_path))
                    continue

                photoprism.update_recommender(photo.id, photo.learned + 1, rank)

            recommender.fit(image_paths, ranks)

            for photo_uid, photo_path in delete_list:
                delete_photo(photo_uid)
                send2trash(photo_path)

        log.info("At present, all photos have been fit")
        recommend_photo_fit_semaphore.release()

    log.warning("The number of threads has reached the maximum")


def recommend_photo_predict():
    photos = photoprism.select_recommender_predict()

    for photo in photos:
        file, photo_path = is_file(photo.files.file_name)

        if not file:
            log.warning("%s is not exist" % photo_path)
            delete_photo(photo.photo_uid)
            continue

        rank = recommender.predict(photo_path)
        photoprism.update_recommender(photo.id, photo.learned + 1, rank)
