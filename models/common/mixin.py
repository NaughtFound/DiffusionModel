import inspect
from typing import Any
import argparse
from copy import copy


class ModelMixin:
    @classmethod
    def _from_kwargs(cls, kwargs: dict[str, Any]):
        sig = inspect.signature(cls.__init__)

        valid_kwargs = {
            key: value for key, value in kwargs.items() if key in sig.parameters
        }

        return cls(**valid_kwargs)

    @classmethod
    def from_params(cls, params: argparse.Namespace):
        return cls._from_kwargs(vars(params))

    @classmethod
    def from_params_with_kwargs(cls, params: argparse.Namespace, **kwargs):
        final_kwargs = copy(vars(params))
        final_kwargs.update(kwargs)

        return cls._from_kwargs(final_kwargs)
