import soundfile as sf

import tensorflow as tf

from tensorflow_tts.inference import TFAutoModel, AutoConfig
from tensorflow_tts.inference import AutoProcessor

txt2mel_conf_fn = r"D:\kidden\mt\open\github\TensorFlowTTS\tensorflow_tts\bin\fastspeech2\conf\fastspeech2.baker.v2.yaml"
print("Loading txt2mel config from", txt2mel_conf_fn)
txt2mel_conf = AutoConfig.from_pretrained(txt2mel_conf_fn)

model_fn = r"D:\dataset\baker\baker\tacotron2\checkpoints\model-12000.h5"
print("Loading txt2mel model from", model_fn)
txt2mel = TFAutoModel.from_pretrained(model_fn, txt2mel_conf)

mel2wav_conf_fn = r"D:\kidden\github\TensorFlowTTS\examples\multiband_melgan\conf\multiband_melgan.baker.v1.yaml"
print("Loading mel2wav config from", mel2wav_conf_fn)
mel2wav_conf = AutoConfig.from_pretrained(mel2wav_conf_fn)

model_fn2 = r"D:\dataset\baker\baker\mbmelgan\checkpoints\generator-80000.h5"
print("Loading mel2wav model from", model_fn2)
mel2wav = TFAutoModel.from_pretrained(model_fn2, mel2wav_conf)

mapper_fn = r"D:\kidden\github\TensorFlowTTS\tensorflow_tts\processor\pretrained\baker_mapper.json"
print("Loading mapper from", mapper_fn)
processor = AutoProcessor.from_pretrained(mapper_fn)

text = "明月依山尽，黄河入海流。"
input_ids = processor.text_to_sequence(text, inference=True)

print("Text to Mel...")
mel_before, mel_after, duration_outputs, _, _ = txt2mel.inference(
    input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
    speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
    speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
    f0_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
    energy_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
)

print("Mel to Wav...")
audio = mel2wav.inference(mel_after)[0, :, 0]

wav_fn = './audio_fs2.wav'
print("Saving wav file into", wav_fn)
sf.write(wav_fn, audio, 22050, "PCM_16")
