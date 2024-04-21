"""Utility functions."""

import importlib


def load_obj_from_config(cfg: dict):
    """Create an object based on the specified module path and kwargs."""
    module_name, class_name = cfg["module"].rsplit(".", maxsplit=1)

    cls = getattr(
        importlib.import_module(module_name),
        class_name,
    )

    return cls(**cfg["kwargs"])


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True
