'''
Date: 2021-01-13 17:31:50
LastEditors: Rustle Karl
LastEditTime: 2021-01-15 19:46:38
'''
import base64
import json
import pickle
from typing import Iterable, Union

# import enlighten
from logger import get_logger
from peewee import (BlobField, BooleanField, CharField, DateTimeField,
                    DoubleField, ForeignKeyField, IntegerField, Model,
                    MySQLDatabase, Proxy, Select)
from playhouse.shortcuts import ReconnectMixin

from client.relation import Relation

log = get_logger("mysql", level="warning")


# manager = enlighten.get_manager()


class ReconnectMySQL(ReconnectMixin, MySQLDatabase, ):
    pass


proxy = Proxy()


class Labels(Model):
    """photoprism.labels"""

    id = IntegerField()
    label_name = CharField()

    class Meta:
        database = proxy
        db_table = "labels"


class Photos(Model):
    """photoprism.photos"""

    id = IntegerField()
    photo_uid = CharField()
    deleted_at = DateTimeField()

    views = IntegerField()  # 浏览次数
    last_visited_at = DateTimeField()  # 最后浏览时间
    like = IntegerField()  # -1, 0, 1
    learned = IntegerField()  # 0 无 1 人脸识别 2 推荐系统 3 全部
    portrait = IntegerField()  # 0 无 1 机器学习认为不是人像 2 机器学习认为是人像
    portrait_id = BlobField()
    photo_rank = DoubleField()  # 推荐系统评分

    class Meta:
        database = proxy
        db_table = "photos"


class Files(Model):
    """photoprism.files"""

    id = IntegerField()
    photo_id = ForeignKeyField(Photos, backref="files")
    file_name = CharField()
    file_root = CharField()
    file_type = CharField()

    class Meta:
        database = proxy
        db_table = "files"

    def __str__(self):
        return json.dumps(self.__dict__["__data__"], ensure_ascii=False)


class Albums(Model):
    """photoprism.albums"""

    id = IntegerField()
    album_uid = CharField()
    deleted_at = DateTimeField()
    portrait = BooleanField()
    portrait_id = BlobField()

    class Meta:
        database = proxy
        db_table = "albums"


class Photoprism(object):
    # deepface_proc = None

    def __init__(self, host="localhost", port=14001):
        self.database = ReconnectMySQL("photoprism", host=host, port=port,
                                       user="photoprism", passwd="photoprism")
        proxy.initialize(self.database)
        self.photos = Photos()
        self.albums = Albums()

        # self.reset_deepface_proc()

    # def reset_deepface_proc(self):
    #     self.deepface_proc = manager.counter(total=self.cnt_deepface_photos(),
    #                                          desc="Deepface", leave=False, color="green3")

    def cnt_deepface_photos(self) -> int:
        return self.photos.select(Photos, Files).join(Files).where(
                (Photos.learned.in_([0, 2])) & (Files.file_root == "/") &
                (Files.file_type.in_(["jpg", "png"])) &
                (Photos.deleted_at == None)).count()  # pylint: disable

    # TODO 分布式
    def select_deepface_photos(self) -> Union[Select, Iterable[Photos]]:
        select = self.photos.select(Photos, Files).join(Files).where(
                (Photos.learned.in_([0, 2])) & (Files.file_root == "/") &
                (Files.file_type.in_(["jpg", "png"])) &
                (Photos.deleted_at == None)).limit(256)  # pylint: disable
        log.debug(select)
        return select

    def select_deepface_albums(self) -> Relation:
        select = self.albums.select().where((Albums.portrait == 1) & (Albums.deleted_at == None))  # pylint: disable
        log.debug(select)
        relation = Relation()
        for row in select:
            row.portrait_id = base64.standard_b64decode(row.portrait_id)
            relation.add(row.album_uid, pickle.loads(row.portrait_id))
        return relation

    def select_recommender_fit(self) -> Union[Select, Iterable[Photos]]:
        select = self.photos.select(Photos, Files).join(Files).where(
                Photos.learned.in_([0, 1]) & Photos.like != 0).limit(200)
        log.debug(select)
        return select

    def select_recommender_predict(self) -> Union[Select, Iterable[Photos]]:
        select = self.photos.select(Photos, Files).join(Files).where(
                Photos.learned.in_([0, 1]) & Photos.like == 0).limit(200)
        log.debug(select)
        return select

    def update_deepface_photos(self, id_: int, learned: int, portrait: int, portrait_id: bytes):
        query = self.photos.update(learned=learned, portrait=portrait,
                                   portrait_id=portrait_id).where(Photos.id == id_)
        log.debug(query)
        query.execute()

        # self.deepface_proc.update()
        # if self.deepface_proc.count == self.deepface_proc.total:
        #     self.reset_deepface_proc()

    def update_recommender(self, id_: int, learned: int, photo_rank: float):
        query = self.photos.update(learned=learned,
                                   photo_rank=photo_rank).where(Photos.id == id_)
        log.debug(query)
        query.execute()
