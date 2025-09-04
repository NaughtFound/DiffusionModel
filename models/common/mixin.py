from typing import Any, Self
import argparse
from copy import deepcopy


class ModelMixin:
    @classmethod
    def _from_kwargs(cls, kwargs: dict[str, Any]) -> Self:
        return cls(**kwargs)

    @classmethod
    def from_params(cls, params: argparse.Namespace) -> Self:
        return cls._from_kwargs(params.__dict__)

    @classmethod
    def from_params_with_kwargs(cls, params: argparse.Namespace, **kwargs) -> Self:
        final_kwargs = deepcopy(params.__dict__)
        final_kwargs.update(kwargs)

        return cls._from_kwargs(final_kwargs)
