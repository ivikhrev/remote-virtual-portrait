"""
Configurations management
"""

import sys
import json
from pathlib import Path

if sys.hexversion < 0x3060000:
    raise Exception("Python version must be >= 3.6")


class ConfigException(Exception):
    """Base configuration exception"""


class Config:
    """Configuration wrapper"""
    _instance = None
    _properties = None

    default_cfg_path = Path(__file__).resolve().parent / "config.json"

    def __new__(cls, *_args, **_kwargs):
        if not Config._instance:
            Config._instance = super(Config, cls).__new__(cls)
        return Config._instance

    def __init__(self, file_path=None, cli_args=None):
        """
        :param file_path: Path to json configuration file
        :type file_path: String
        """
        if Config._properties:
            return

        self._file_path = file_path or Config.default_cfg_path
        self._load_cfg()
        Config._properties = {}

        for name, value in self._json_cfg.items():
            Config._properties[name] = value

    def _load_cfg(self):
        """Load the json configuration file"""
        try:
            with open(self._file_path, encoding="utf-8") as conf:
                self._json_cfg = json.load(conf)
        except Exception as exc:
            raise ConfigException("Failed to load configuration from:", self._file_path) from exc

    @property
    def properties(self):
        """Get all properties as Dict"""
        return self._properties