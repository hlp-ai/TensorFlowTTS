"""Tensorflow Auto Processor modules."""

import logging
import json
import os
from collections import OrderedDict

from tensorflow_tts.processor import (
    LJSpeechProcessor,
    BakerProcessor,
    LibriTTSProcessor,
    LJSpeechUltimateProcessor,
)

from tensorflow_tts.utils import CACHE_DIRECTORY, PROCESSOR_FILE_NAME, LIBRARY_NAME
from tensorflow_tts import __version__ as VERSION
from huggingface_hub import hf_hub_url, cached_download

CONFIG_MAPPING = OrderedDict(
    [
        ("LJSpeechProcessor", LJSpeechProcessor),
        ("BakerProcessor", BakerProcessor),
        ("LibriTTSProcessor", LibriTTSProcessor),
        ("LJSpeechUltimateProcessor", LJSpeechUltimateProcessor),
    ]
)


class AutoProcessor:
    def __init__(self):
        raise EnvironmentError(
            "AutoProcessor is designed to be instantiated "
            "using the `AutoProcessor.from_pretrained(pretrained_path)` method."
        )

    @classmethod
    def from_pretrained(cls, pretrained_path, **kwargs):
        # load weights from hf hub
        if not os.path.isfile(pretrained_path):
            # retrieve correct hub url
            download_url = hf_hub_url(repo_id=pretrained_path, filename=PROCESSOR_FILE_NAME)

            pretrained_path = str(
                cached_download(
                    url=download_url,
                    library_name=LIBRARY_NAME,
                    library_version=VERSION,
                    cache_dir=CACHE_DIRECTORY,
                )
            )
        with open(pretrained_path, "r") as f:
            config = json.load(f)

        try:
            processor_name = config["processor_name"]
            processor_class = CONFIG_MAPPING[processor_name]
            processor_class = processor_class(
                data_dir=None, loaded_mapper_path=pretrained_path
            )
            return processor_class
        except Exception:
            raise ValueError(
                "Unrecognized processor in {}. "
                "Should have a `processor_name` key in its config.json, or contain one of the following strings "
                "in its name: {}".format(
                    pretrained_path, ", ".join(CONFIG_MAPPING.keys())
                )
            )
