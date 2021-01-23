import threading


class Relation(object):
    __list = []
    __index = {}

    __lock = threading.RLock()
    __next = -1

    def add(self, key, value):
        with self.__lock:
            self.__index[len(self.__list)] = key
            self.__list.append(value)

    def key(self, index: int):
        with self.__lock:
            return self.__index.get(index)

    def values(self):
        with self.__lock:
            return self.__list
