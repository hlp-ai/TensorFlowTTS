"""Tensorflow Auto Config modules."""

import yaml
import os
from collections import OrderedDict

from tensorflow_tts.configs import (
    FastSpeechConfig,
    FastSpeech2Config,
    MelGANGeneratorConfig,
    MultiBandMelGANGeneratorConfig,
    Tacotron2Config,
)

from tensorflow_tts.utils import CACHE_DIRECTORY, CONFIG_FILE_NAME, LIBRARY_NAME
from tensorflow_tts import __version__ as VERSION
from huggingface_hub import hf_hub_url, cached_download

CONFIG_MAPPING = OrderedDict(
    [
        ("fastspeech", FastSpeechConfig),
        ("fastspeech2", FastSpeech2Config),
        ("multiband_melgan_generator", MultiBandMelGANGeneratorConfig),
        ("melgan_generator", MelGANGeneratorConfig),
        ("tacotron2", Tacotron2Config),
    ]
)


class AutoConfig:
    def __init__(self):
        raise EnvironmentError(
            "AutoConfig is designed to be instantiated "
            "using the `AutoConfig.from_pretrained(pretrained_path)` method."
        )

    @classmethod
    def from_pretrained(cls, pretrained_path, **kwargs):
        # load weights from hf hub
        if not os.path.isfile(pretrained_path):
            # retrieve correct hub url
            download_url = hf_hub_url(
                repo_id=pretrained_path, filename=CONFIG_FILE_NAME
            )

            pretrained_path = str(
                cached_download(
                    url=download_url,
                    library_name=LIBRARY_NAME,
                    library_version=VERSION,
                    cache_dir=CACHE_DIRECTORY,
                )
            )

        with open(pretrained_path) as f:
            config = yaml.load(f, Loader=yaml.Loader)

        try:
            model_type = config["model_type"]
            config_class = CONFIG_MAPPING[model_type]
            config_class = config_class(**config[model_type + "_params"], **kwargs)
            config_class.set_config_params(config)
            return config_class
        except Exception:
            raise ValueError(
                "Unrecognized config in {}. "
                "Should have a `model_type` key in its config.yaml, or contain one of the following strings "
                "in its name: {}".format(
                    pretrained_path, ", ".join(CONFIG_MAPPING.keys())
                )
            )
