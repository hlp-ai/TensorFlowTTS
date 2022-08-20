import os
import re
import tempfile
import wave

import pyaudio
import soundfile
import tensorflow as tf
from langid import langid

from tensorflow_tts.inference import AutoProcessor, TFAutoModel


class Tacotron2MBGan:

    def __init__(self, processor, tacotron2, mb_melgan):
        self.processor = AutoProcessor.from_pretrained(processor)
        self.tacotron2 = TFAutoModel.from_pretrained(tacotron2)
        self.mb_melgan = TFAutoModel.from_pretrained(mb_melgan)

    def _towav(self, text):
        input_ids = self.processor.text_to_sequence(text, inference=True)

        # tacotron2 inference (text-to-mel)
        decoder_output, mel_outputs, stop_token_prediction, alignment_history = self.tacotron2.inference(
            input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            input_lengths=tf.convert_to_tensor([len(input_ids)], tf.int32),
            speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
        )

        # melgan inference (mel-to-wav)
        audio = self.mb_melgan.inference(mel_outputs)[0, :, 0]

        return audio

    def towav(self, text):
        return self._towav(text)


def play_wav(fn):
    CHUNK = 1024
    wf = wave.open(fn, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(CHUNK)

    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(CHUNK)

    stream.stop_stream()
    stream.close()
    p.terminate()


def split_text(text):
    return re.split(r"[.?,:。？，：]", text)


synthesizers = {"en": Tacotron2MBGan("tensorspeech/tts-tacotron2-ljspeech-en",
                                "tensorspeech/tts-tacotron2-ljspeech-en",
                                "tensorspeech/tts-mb_melgan-ljspeech-en"),
               "zh": Tacotron2MBGan("tensorspeech/tts-tacotron2-baker-ch",
                                "tensorspeech/tts-tacotron2-baker-ch",
                                "tensorspeech/tts-mb_melgan-baker-ch")}


if __name__ == "__main__":
    while True:
        text = input("输入要合成语音的文本: ")

        lang = langid.classify(text)[0]
        synthesizer = synthesizers[lang]

        print("Generating wav...")
        wav = synthesizer.towav(text)
        # sentences = split_text(text)
        # total_wav = []
        # for s in sentences:
        #     wav = synthesizer.towav(s)
        #     print(wav)
        #     total_wav.append(wav)

        print("Saving wav...")
        tmp_wav_fn = os.path.join(tempfile.gettempdir(), str(hash(text)) + ".wav")
        # soundfile.write(tmp_wav_fn, tf.concat(total_wav, axis=0), 22050, "PCM_16")
        soundfile.write(tmp_wav_fn, wav, 22050, "PCM_16")

        print("Playing wav...")
        play_wav(tmp_wav_fn)
