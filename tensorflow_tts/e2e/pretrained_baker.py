import soundfile as sf

import tensorflow as tf

from tensorflow_tts.inference import AutoProcessor
from tensorflow_tts.inference import TFAutoModel

processor = AutoProcessor.from_pretrained("tensorspeech/tts-tacotron2-baker-ch")
tacotron2 = TFAutoModel.from_pretrained("tensorspeech/tts-tacotron2-baker-ch")
mb_melgan = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-baker-ch")

text = "这是一个开源的端到端中文语音合成系统"

input_ids = processor.text_to_sequence(text, inference=True)

# tacotron2 inference (text-to-mel)
decoder_output, mel_outputs, stop_token_prediction, alignment_history = tacotron2.inference(
    input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
    input_lengths=tf.convert_to_tensor([len(input_ids)], tf.int32),
    speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
)

# melgan inference (mel-to-wav)
audio = mb_melgan.inference(mel_outputs)[0, :, 0]

# save to file
sf.write('./audio_baker.wav', audio, 22050, "PCM_16")
