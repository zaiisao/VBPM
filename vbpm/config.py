"""The mainline config schema, loaded from ``config_schema.json``."""
from __future__ import annotations

import importlib
import json
import pathlib
from types import SimpleNamespace

import yaml

SCHEMA_PATH = pathlib.Path(__file__).with_name("config_schema.json")

REQUIRED_FIELDS = {"type", "default", "description"}
OPTIONAL_FIELDS = {"enum", "minimum"}

# JSON Schema type names -> the Python check. bool is excluded from the number types
# because True would otherwise pass as 1 and quietly become a learning rate.
TYPES = {
    "boolean": lambda v: isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "string": lambda v: isinstance(v, str),
    "array": lambda v: isinstance(v, (list, tuple)),
}


def check(key: str, value, spec: dict) -> None:
    """Refuse a value its spec does not admit (type, then enum, then bound)."""
    kind = spec["type"]
    assert TYPES[kind](value), f"config key {key!r}: expected {kind}, got {value!r}"
    choices = spec.get("enum")
    assert choices is None or value in choices, \
        f"config key {key!r}: {value!r} is not one of {choices}"
    low = spec.get("minimum")
    assert low is None or value >= low, \
        f"config key {key!r}: {value!r} is below the minimum {low}"


def schema() -> dict:
    """The mainline table, checked for well-formedness as it is read."""
    loaded = json.loads(SCHEMA_PATH.read_text())
    for key, spec in loaded.items():
        missing = REQUIRED_FIELDS - set(spec)
        assert not missing, f"schema key {key!r} is missing {sorted(missing)}"
        extra = set(spec) - REQUIRED_FIELDS - OPTIONAL_FIELDS
        assert not extra, f"schema key {key!r} has unknown spec fields {sorted(extra)}"
        assert spec["type"] in TYPES, f"schema key {key!r} has type {spec['type']!r}"
        check(key, spec["default"], spec)      # a default must satisfy its own spec
    return loaded


def load_config(path: str, overrides: list[str] = ()):
    """YAML + --set overrides -> (cfg namespace, hooks module)."""
    recipe = yaml.safe_load(pathlib.Path(path).read_text()) or {}
    for item in overrides:
        key, _, value = item.partition("=")
        recipe[key.strip().replace("-", "_")] = yaml.safe_load(value)

    mainline = schema()
    hooks = importlib.import_module("vbpm.model")
    known = {key: spec["default"] for key, spec in mainline.items()}
    known |= hooks.DEFAULTS
    unknown = set(recipe) - set(known)
    assert not unknown, f"unknown config keys {sorted(unknown)} (typo?)"
    for key, value in recipe.items():
        if key in mainline:
            check(key, value, mainline[key])
    return SimpleNamespace(**(known | recipe)), hooks
