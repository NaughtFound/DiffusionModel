from functools import wraps
from collections.abc import Callable


class KWargs(object):
    def __new__(cls):
        """creates a singleton object, if it is not created,
        or else returns the previous singleton object"""
        if not hasattr(cls, "instance"):
            cls.instance = super(KWargs, cls).__new__(cls)
        return cls.instance

    def get(self, key: str) -> dict:
        kwargs = {}

        if hasattr(self, key):
            kwargs = getattr(self, key)

        return kwargs

    def insert(self, f: Callable, **kwargs):
        key = f.__qualname__

        setattr(self, key, kwargs)


def with_kwargs(func: Callable):
    key = func.__qualname__

    @wraps(func)
    def wrapper(*args, **kwargs):
        additional_kwargs = KWargs().get(key)

        return func(*args, **kwargs, **additional_kwargs)

    return wrapper
