import logging
import os

import pytest

from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import AutoProcessor
from tensorflow_tts.inference import TFAutoModel

os.environ["CUDA_VISIBLE_DEVICES"] = ""

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)


@pytest.mark.parametrize(
    "mapper_path", 
    [
        "./files/baker_mapper.json",
        "./files/libritts_mapper.json",
        "./files/ljspeech_mapper.json",
     ]
)
def test_auto_processor(mapper_path):
    processor = AutoProcessor.from_pretrained(pretrained_path=mapper_path)
    processor.save_pretrained("./test_saved")
    processor = AutoProcessor.from_pretrained("./test_saved/processor.json")


@pytest.mark.parametrize(
    "config_path", 
    [
        "../tensorflow_tts/bin/fastspeech2/conf/fastspeech2.v1.yaml",
        "../tensorflow_tts/bin/fastspeech2/conf/fastspeech2.v2.yaml",
        "../tensorflow_tts/bin/mbmelgan/conf/multiband_melgan.v1.yaml",
        "../tensorflow_tts/bin/tacotron2/conf/tacotron2.v1.yaml",
     ]
)
def test_auto_model(config_path):
    config = AutoConfig.from_pretrained(pretrained_path=config_path)
    model = TFAutoModel.from_pretrained(pretrained_path=None, config=config)

    # test save_pretrained
    config.save_pretrained("./test_saved")
    model.save_pretrained("./test_saved")

    # test from_pretrained
    config = AutoConfig.from_pretrained("./test_saved/config.yml")
    model = TFAutoModel.from_pretrained("./test_saved/model.h5", config=config)
