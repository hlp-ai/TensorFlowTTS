#!/usr/bin/env python
# coding: utf-8
# 
# -----
# Change logs
# * 2020-07-04 KST : Update notebook with the lastest TensorflowTTS repo.
#  * compatible with https://github.com/TensorSpeech/TensorflowTTS/pull/83
# * 2020-07-02 KST : Third implementation (outputs : `tacotron2.tflite`) 
#  * **varied-length** input tensor, **varied-length** output tensor
# -----
# 
# **Status** : successfully converted (`tacotron2.tflite`)
# 
# **Disclaimer** 
# -  This colab doesn't care about the latency, so it compressed the model with quantization. (129 MB -> 33 MB)
# - The TFLite file doesn't have LJSpeechProcessor. So you need to run it before feeding input vectors.
# - `tf-nightly>=2.4.0-dev20200630`
# 

# # Generate voice with Tacotron2

# !pip install tf-nightly


import tensorflow as tf

from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.processor import LJSpeechProcessor

print(tf.__version__)

# initialize melgan model
melgan = TFAutoModel.from_pretrained("tensorspeech/tts-melgan-ljspeech-en")

# initialize Tacotron2 model.
tacotron2 = TFAutoModel.from_pretrained("tensorspeech/tts-tacotron2-ljspeech-en", enable_tflite_convertible=True)

# Newly added :
tacotron2.setup_window(win_front=6, win_back=6)
tacotron2.setup_maximum_iterations(3000)

tacotron2.summary()


# # Convert to TF Lite

# Concrete Function
tacotron2_concrete_function = tacotron2.inference_tflite.get_concrete_function()


converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [tacotron2_concrete_function]
)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

# Save the TF Lite model.
with open('tacotron2.tflite', 'wb') as f:
  f.write(tflite_model)

print('Model size is %f MBs.' % (len(tflite_model) / 1024 / 1024.0) )


# # Inference from TFLite


import numpy as np
import tensorflow as tf

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path='tacotron2.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Prepare input data.
def prepare_input(input_ids):
  return (tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
          tf.convert_to_tensor([len(input_ids)], tf.int32),
          tf.convert_to_tensor([0], dtype=tf.int32))


# Test the model on random input data.
def infer(input_text):
  processor = LJSpeechProcessor(None, "english_cleaners")
  input_ids = processor.text_to_sequence(input_text.lower())
  input_ids = np.concatenate([input_ids, [len(symbols) - 1]], -1)  # eos.
  interpreter.resize_tensor_input(input_details[0]['index'], 
                                  [1, len(input_ids)])
  interpreter.allocate_tensors()
  input_data = prepare_input(input_ids)
  for i, detail in enumerate(input_details):
    print(detail)
    input_shape = detail['shape']
    interpreter.set_tensor(detail['index'], input_data[i])

  interpreter.invoke()

  # The function `get_tensor()` returns a copy of the tensor data.
  # Use `tensor()` in order to get a pointer to the tensor.
  return (interpreter.get_tensor(output_details[0]['index']),
          interpreter.get_tensor(output_details[1]['index']))


input_text = "Recent research at Harvard has shown meditating\
for as little as 8 weeks, can actually increase the grey matter in the \
parts of the brain responsible for emotional regulation, and learning."

decoder_output_tflite, mel_output_tflite = infer(input_text)
audio_before_tflite = melgan(decoder_output_tflite)[0, :, 0]
audio_after_tflite = melgan(mel_output_tflite)[0, :, 0]
