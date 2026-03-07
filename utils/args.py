from collections.abc import Callable
from functools import partial, wraps
from typing import Self


class KWargs:
    def __new__(cls) -> Self:
        if not hasattr(cls, "instance"):
            cls.instance = super().__new__(cls)
        return cls.instance

    @staticmethod
    def get(key: str) -> dict:
        kw = KWargs()
        kwargs = {}

        if hasattr(kw, key):
            kwargs = getattr(kw, key)

        return kwargs

    @staticmethod
    def drop(func: Callable) -> None:
        kw = KWargs()
        key = func.__qualname__

        if hasattr(kw, key):
            delattr(kw, key)

    @staticmethod
    def insert(func: Callable, **kwargs) -> None:
        kw = KWargs()
        key = func.__qualname__

        setattr(kw, key, kwargs)


def with_kwargs[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    key = func.__qualname__

    @wraps(func)
    def wrapper(*args, **kwargs) -> R:
        additional_kwargs = KWargs.get(key)

        fn = partial(func, **additional_kwargs)

        return fn(*args, **kwargs)

    return wrapper
