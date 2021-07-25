from typing import TypeVar, Generic

T = TypeVar('T')


class MultivalueMap(Generic[T]):

    def __init__(self):
        self.__dict = {}

    def __getitem__(self, key: str):
        return self.__dict[key]

    def __setitem__(self, key: str, value: T):
        if key in dict:
            self.__dict[key].append(value)
        else:
            self.__dict[key] = [value]