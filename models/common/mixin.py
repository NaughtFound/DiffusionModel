from typing import Self
import argparse


class ModelMixin:
    @classmethod
    def from_params(cls, params: argparse.Namespace) -> Self:
        return cls(**params.__dict__)
