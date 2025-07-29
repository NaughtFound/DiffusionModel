import importlib
from importlib.util import spec_from_file_location, module_from_spec
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Optional, Union
import yaml


class ConfigParser:
    config: dict[str, Any]
    local: Optional[ModuleType]
    variables: dict[str, Any]

    def __init__(self, config_path: Union[str, Path]):
        root, _ = os.path.split(config_path)

        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        if "local" in self.config:
            local_path = self.config.pop("local")
            self.local = self._load_python_module(os.path.join(root, local_path))
        else:
            self.local = None

        self.variables = {}

    def _load_python_module(self, path: str) -> ModuleType:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Local Python file not found: {path}")

        module_name = os.path.splitext(os.path.basename(path))[0]
        spec = spec_from_file_location(module_name, path)
        module = module_from_spec(spec)
        loader = spec.loader
        if loader is None:
            raise ImportError(f"Failed to load module from: {path}")
        loader.exec_module(module)
        return module

    def _load_object(self, module_path: str, object_name: str) -> Any:
        try:
            module = importlib.import_module(module_path)
            if not hasattr(module, object_name):
                raise AttributeError(
                    f"Module '{module_path}' has no attribute '{object_name}'"
                )
            return getattr(module, object_name)
        except ModuleNotFoundError as e:
            raise ImportError(f"Could not import module '{module_path}': {e}") from e

    def _resolve_entry(self, entry: Any) -> Any:
        if isinstance(entry, str) and entry.startswith("args."):
            key = entry.split(".")[1]
            entry = self.variables.get(key)

            return entry

        if not isinstance(entry, dict):
            return entry

        module_path = entry["module"]
        symbol_name = entry["source"]
        call = entry.get("call", True)

        obj = None

        if module_path == "local":
            obj = getattr(self.local, symbol_name)
        else:
            obj = self._load_object(module_path, symbol_name)

        if callable(obj) and call:
            args = entry.get("args", {})

            if isinstance(args, dict):
                for k in args:
                    args[k] = self._resolve_entry(args[k])
                    self.variables[k] = args[k]

            if isinstance(call, bool):
                return obj(**args)

            if not hasattr(obj, call):
                raise AttributeError(
                    f"Module '{symbol_name}' has no attribute '{call}'"
                )

            fn = getattr(obj, call)

            if not callable(fn):
                raise TypeError(f"{fn} is not callable")

            return fn(**args)

        return obj

    def parse(self) -> dict[str, Any]:
        res = {}

        for k in self.config:
            res[k] = self._resolve_entry(self.config[k])

        return res
