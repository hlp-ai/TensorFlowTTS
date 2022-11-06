import tensorflow as tf

from tensorflow_tts.inference import AutoProcessor
from tensorflow_tts.inference import TFAutoModel

print(tf.__version__) # check if >= 2.4.0

# initialize melgan model
melgan = TFAutoModel.from_pretrained("tensorspeech/tts-melgan-ljspeech-en")

# initialize FastSpeech model.
fastspeech = TFAutoModel.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en")

input_text = "Recent research at Harvard has shown meditating\
for as little as 8 weeks, can actually increase the grey matter in the \
parts of the brain responsible for emotional regulation, and learning."

processor = AutoProcessor.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en")
input_ids = processor.text_to_sequence(input_text.lower())

mel_before, mel_after, duration_outputs, _, _ = fastspeech.inference(
    input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
    speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
    speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
    f0_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
    energy_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
)

audio_before = melgan(mel_before)[0, :, 0]
audio_after = melgan(mel_after)[0, :, 0]

# # Convert to TFLite

# Concrete Function
fastspeech_concrete_function = fastspeech.inference_tflite.get_concrete_function()

converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [fastspeech_concrete_function]
)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

# Save the TF Lite model.
with open('fastspeech_quant.tflite', 'wb') as f:
  f.write(tflite_model)

print('Model size is %f MBs.' % (len(tflite_model) / 1024 / 1024.0) )


# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path='fastspeech_quant.tflite')

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input data.
def prepare_input(input_ids):
  input_ids = tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0)
  return (input_ids,
          tf.convert_to_tensor([0], tf.int32),
          tf.convert_to_tensor([1.0], dtype=tf.float32),
          tf.convert_to_tensor([1.0], dtype=tf.float32),
          tf.convert_to_tensor([1.0], dtype=tf.float32))

# Test the model on random input data.
def infer(input_text):
  processor = AutoProcessor.from_pretrained(pretrained_path=r"D:\kidden\github\TensorFlowTTS\tensorflow_tts\processor\pretrained\ljspeech_mapper.json")
  input_ids = processor.text_to_sequence(input_text.lower())
  interpreter.resize_tensor_input(input_details[0]['index'], 
                                  [1, len(input_ids)])
  interpreter.resize_tensor_input(input_details[1]['index'], 
                                  [1])
  interpreter.resize_tensor_input(input_details[2]['index'], 
                                  [1])
  interpreter.resize_tensor_input(input_details[3]['index'], 
                                  [1])
  interpreter.resize_tensor_input(input_details[4]['index'], 
                                  [1])
  interpreter.allocate_tensors()
  input_data = prepare_input(input_ids)
  for i, detail in enumerate(input_details):
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
