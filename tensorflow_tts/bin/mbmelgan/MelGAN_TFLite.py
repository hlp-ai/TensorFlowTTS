#!/usr/bin/env python
# coding: utf-8

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tulasiram58827/TTS_TFLite/blob/main/MelGAN_TFLite.ipynb)

# This notebook contains code to convert TensorFlow MelGAN to TFLite

# ## Acknowledgments

# - Pretrained model downloaded from [TensorFlowTTS Repository](https://github.com/TensorSpeech/TensorFlowTTS/tree/master/examples/melgan#pretrained-models-and-audio-samples)
# 
# - Most of the code is inspired from [TensorFlowTTS Repository](https://github.com/TensorSpeech/TensorFlowTTS/)

# ## Imports

import tensorflow as tf
import yaml


from tensorflow_tts.configs import MelGANGeneratorConfig
from tensorflow_tts.models import TFMelGANGenerator

import numpy as np

from IPython.display import Audio


# ## Download Model and Config

# # Download Model
# get_ipython().system('gdown --id 1AKEx1NoVhHH2EaHCCZbHWeIF_U8UCGGJ -O model.h5')
#
# # Download Config
# get_ipython().system('wget https://raw.githubusercontent.com/TensorSpeech/TensorFlowTTS/master/examples/melgan/conf/melgan.v1.yaml')


# ## Load Model


with open('/content/melgan.v1.yaml') as f:
    config = yaml.load(f, Loader=yaml.Loader)

melgan = TFMelGANGenerator(
        config=MelGANGeneratorConfig(**config["melgan_generator_params"]), name="melgan_generator")
melgan._build()
melgan.load_weights('model.h5')


# ## Convert to TFLite


def convert_to_tflite(quantization):
    melgan_concrete_function = melgan.inference_tflite.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([melgan_concrete_function])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
    if quantization == 'float16':
        converter.target_spec.supported_types = [tf.float16]
    tf_lite_model = converter.convert()
    model_name = f'melgan_{quantization}.tflite'
    with open(model_name, 'wb') as f:
      f.write(tf_lite_model)


# #### Dynamic Range Quantization


quantization = 'dr' #@param ["dr", "float16"]
convert_to_tflite(quantization)
# get_ipython().system('du -sh melgan_dr.tflite')


# #### Float16 Quantization

quantization = 'float16' #@param ["dr", "float16"]
convert_to_tflite(quantization)
# get_ipython().system('du -sh melgan_float16.tflite')


# ## Download Sample Output of Tacotron2


# get_ipython().system('gdown --id 1LmU3j8yedwBzXKVDo9tCvozLM4iwkRnP -O tac_output.npy')


# ## TFLite Inference

data = np.load('tac_output.npy')
feats = np.expand_dims(data, 0)

interpreter = tf.lite.Interpreter(model_path='melgan_dr.tflite')

input_details = interpreter.get_input_details()

output_details = interpreter.get_output_details()

interpreter.resize_tensor_input(input_details[0]['index'],  [1, feats.shape[1], feats.shape[2]], strict=True)
interpreter.allocate_tensors()

interpreter.set_tensor(input_details[0]['index'], feats)

interpreter.invoke()

output = interpreter.get_tensor(output_details[0]['index'])


# ## Play Audio


output = np.squeeze(output)
Audio(output, rate=22050)

