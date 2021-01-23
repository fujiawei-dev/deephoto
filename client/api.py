'''
Date: 2021-01-14 17:32:10
LastEditors: Rustle Karl
LastEditTime: 2021-01-21 15:22:53
'''

import base64
import pickle
from datetime import datetime

import numpy as np
import requests
from logger import get_logger

log = get_logger("api", level="info")

Host = "localhost"
Port = 12342
Prefix = "api/v1/"
Url = "http://%s:%d/%s" % (Host, Port, Prefix)

Headers = {"User-Agent": "Deephoto/1.0"}


# DELETE /api/v1/photos/:uid?permanently=true
def delete_photo(photo_uid: str) -> requests.Response:
    log.debug("delete_photo(%s)" % photo_uid)
    return requests.delete(Url + "photos/%s?permanently=true" % photo_uid, headers=Headers)


# POST /api/v1/albums
def create_album(portrait_id: np.ndarray) -> requests.Response:
    now = datetime.now()
    json = {
        "Title": "Unknown-%s" % now.strftime("%Y%m%d%H%M%S"),
        "Description": "This album was auto-created by Deephoto on %s"
                       % now.strftime("%Y.%m.%d %H:%M:%S"),
        "Notes": "auto",
        "Type": "album",  # people
        "Category": "People",
        "Year": now.year,
        "Month": now.month,
        "Day": now.day,
        "Portrait": True,
        "PortraitID": base64.standard_b64encode(pickle.dumps(portrait_id)).decode("utf-8"),
    }
    log.info("Create: %s" % json["Title"])
    res = requests.post(Url + "albums", json=json, headers=Headers)
    if res.status_code != 200:
        return False
    return res.json()["UID"]


# POST /api/v1/albums/:uid/photos
def add_photos_to_album(album_uid: str, *photos_uid: str) -> requests.Response:
    json = {"album_uid": album_uid, "photos": photos_uid}
    log.info("Add: %s " % json)
    return requests.post(Url + "albums/%s/photos" % album_uid, json=json, headers=Headers)


if __name__ == "__main__":
    create_album("xxx")