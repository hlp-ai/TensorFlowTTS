import soundfile as sf
import numpy as np

import tensorflow as tf

from tensorflow_tts.inference import AutoProcessor
from tensorflow_tts.inference import TFAutoModel

processor = AutoProcessor.from_pretrained("tensorspeech/tts-fastspeech2-baker-ch")
fastspeech2 = TFAutoModel.from_pretrained("tensorspeech/tts-fastspeech2-baker-ch")

mb_melgan = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-baker-ch")

text = "这是一个开源的端到端中文语音合成系统"

input_ids = processor.text_to_sequence(text, inference=True)

mel_before, mel_after, duration_outputs, _, _ = fastspeech2.inference(
    input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
    speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
    speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
    f0_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
    energy_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
)

# melgan inference (mel-to-wav)
audio = mb_melgan.inference(mel_after)[0, :, 0]

# save to file
sf.write('./audio_baker_fs2.wav', audio, 22050, "PCM_16")
