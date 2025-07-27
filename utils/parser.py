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

    def __init__(self, config_path: Union[str, Path]):
        try:
            with open(config_path, "r") as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Error: The file '{config_path}' was not found.")
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")

        if "local" in self.config:
            local_path = self.config.pop("local")
            self.local = self._load_python_module(local_path)
        else:
            self.local = None

    def _load_python_module(path: str) -> ModuleType:
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

    def _load_object(module_path: str, object_name: str) -> Any:
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
        if not isinstance(entry, dict):
            return entry

        module_path = entry["module"]
        symbol_name = entry["source"]
        args = entry.get("args", {})

        obj = None

        if module_path == "local":
            obj = getattr(self.local, symbol_name)
        else:
            obj = self._load_object(module_path, symbol_name)

        for k in args:
            args[k] = self._resolve_entry(args[k])

        return obj(**args) if callable(obj) and args else obj

    def parse(self) -> dict[str, Any]:
        res = {}

        for k in self.config:
            res[k] = self._resolve_entry(self.config[k])

        return res
