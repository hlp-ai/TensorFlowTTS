
import tensorflow as tf

from tensorflow_tts.inference import TFAutoModel, AutoConfig, AutoProcessor

print(tf.__version__)

txt2mel_conf_fn = r"D:\kidden\github\TensorFlowTTS\examples\tacotron2\conf\tacotron2.baker.v1.yaml"
print("Loading txt2mel config from", txt2mel_conf_fn)
txt2mel_conf = AutoConfig.from_pretrained(txt2mel_conf_fn)

model_fn = r"D:\dataset\baker\baker\tacotron2\checkpoints\model-12000.h5"
print("Loading txt2mel model from", model_fn)
tacotron2 = TFAutoModel.from_pretrained(model_fn, txt2mel_conf)

# tacotron2.setup_window(win_front=6, win_back=6)
# tacotron2.setup_maximum_iterations(3000)

tacotron2.summary()


# # Convert to TF Lite

# Concrete Function
tacotron2_concrete_function = tacotron2.inference_tflite.get_concrete_function()


converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [tacotron2_concrete_function]
)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

# Save the TF Lite model.
with open('tacotron2.tflite', 'wb') as f:
  f.write(tflite_model)

print('Model size is %f MBs.' % (len(tflite_model) / 1024 / 1024.0) )


# # Inference from TFLite

mapper_fn = r"D:\kidden\github\TensorFlowTTS\tensorflow_tts\processor\pretrained\baker_mapper.json"
print("Loading mapper from", mapper_fn)
processor = AutoProcessor.from_pretrained(mapper_fn)

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
  input_ids = processor.text_to_sequence(input_text.lower(), inference=True)
  # input_ids = np.concatenate([input_ids, [len(symbols) - 1]], -1)  # eos.
  interpreter.resize_tensor_input(input_details[0]['index'], [1, len(input_ids)])
  interpreter.allocate_tensors()
  input_data = prepare_input(input_ids)
  for i, detail in enumerate(input_details):
    print(detail)
    interpreter.set_tensor(detail['index'], input_data[i])

  interpreter.invoke()

  # The function `get_tensor()` returns a copy of the tensor data.
  # Use `tensor()` in order to get a pointer to the tensor.
  return (interpreter.get_tensor(output_details[0]['index']),
          interpreter.get_tensor(output_details[1]['index']))


input_text = "这是中文合成测试。"

decoder_output_tflite, mel_output_tflite = infer(input_text)

print(mel_output_tflite)
