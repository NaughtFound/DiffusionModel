from functools import wraps
from collections.abc import Callable


class KWargs(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(KWargs, cls).__new__(cls)
        return cls.instance

    @staticmethod
    def get(key: str) -> dict:
        kw = KWargs()
        kwargs = {}

        if hasattr(kw, key):
            kwargs = getattr(kw, key)

        return kwargs

    @staticmethod
    def drop(func: Callable):
        kw = KWargs()
        key = func.__qualname__

        if hasattr(kw, key):
            delattr(kw, key)

    @staticmethod
    def insert(func: Callable, **kwargs):
        kw = KWargs()
        key = func.__qualname__

        setattr(kw, key, kwargs)


def with_kwargs(func: Callable):
    key = func.__qualname__

    @wraps(func)
    def wrapper(*args, **kwargs):
        additional_kwargs = KWargs.get(key)

        return func(*args, **kwargs, **additional_kwargs)

    return wrapper
