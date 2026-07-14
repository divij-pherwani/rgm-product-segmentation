"""Load config.yaml.

Everything the pipeline needs to know lives in that one file. Nothing is hard-coded in the
code — if you want to change a threshold, change it there, not here.
"""
from pathlib import Path
import yaml


class Settings(dict):
    """A dict you can read with dots: settings.grouping.random_seed"""

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError:
            raise AttributeError(f"'{name}' isn't in config.yaml")
        return Settings(value) if isinstance(value, dict) else value


def load(path="config.yaml") -> Settings:
    settings = Settings(yaml.safe_load(Path(path).read_text()))

    # Fail here, with a clear message, rather than three steps later with a confusing one
    for section in ["data", "cleaning", "features", "grouping", "checks", "relationships"]:
        if section not in settings:
            raise ValueError(f"config.yaml is missing the '{section}' section")
    return settings
