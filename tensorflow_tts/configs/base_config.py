"""Base Config for all config."""

import abc
import yaml
import os

from tensorflow_tts.utils.utils import CONFIG_FILE_NAME


class BaseConfig(abc.ABC):
    def set_config_params(self, config_params):
        self.config_params = config_params

    def save_pretrained(self, saved_path):
        """Save config to file"""
        os.makedirs(saved_path, exist_ok=True)
        with open(os.path.join(saved_path, CONFIG_FILE_NAME), "w") as file:
            yaml.dump(self.config_params, file, Dumper=yaml.Dumper)
