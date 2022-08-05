"""Base Model for all model."""

import tensorflow as tf
import yaml
import os
import numpy as np

from tensorflow_tts.utils.utils import MODEL_FILE_NAME, CONFIG_FILE_NAME


class BaseModel(tf.keras.Model):
    def set_config(self, config):
        self.config = config

    def save_pretrained(self, saved_path):
        """Save config and weights to file"""
        os.makedirs(saved_path, exist_ok=True)
        self.config.save_pretrained(saved_path)
        self.save_weights(os.path.join(saved_path, MODEL_FILE_NAME))
