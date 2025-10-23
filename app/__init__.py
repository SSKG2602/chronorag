from importlib import import_module
from typing import Any


def create_app(*args: Any, **kwargs: Any):
    module = import_module("app.main")
    return module.create_app(*args, **kwargs)


__all__ = ["create_app"]
