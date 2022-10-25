"""Tensorflow Savable Model modules."""

import tensorflow as tf

from tensorflow_tts.models import (
    TFFastSpeech,
    TFFastSpeech2,
    TFMelGANGenerator,
    TFMBMelGANGenerator,
    TFTacotron2,
)


class SavableTFTacotron2(TFTacotron2):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def call(self, inputs, training=False):
        input_ids, input_lengths, speaker_ids = inputs
        return super().inference(input_ids, input_lengths, speaker_ids)

    def _build(self):
        input_ids = tf.convert_to_tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=tf.int32)
        input_lengths = tf.convert_to_tensor([9], dtype=tf.int32)
        speaker_ids = tf.convert_to_tensor([0], dtype=tf.int32)
        self([input_ids, input_lengths, speaker_ids])


class SavableTFFastSpeech(TFFastSpeech):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def call(self, inputs, training=False):
        input_ids, speaker_ids, speed_ratios = inputs
        return super()._inference(input_ids, speaker_ids, speed_ratios)

    def _build(self):
        input_ids = tf.convert_to_tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], tf.int32)
        speaker_ids = tf.convert_to_tensor([0], tf.int32)
        speed_ratios = tf.convert_to_tensor([1.0], tf.float32)
        self([input_ids, speaker_ids, speed_ratios])


class SavableTFFastSpeech2(TFFastSpeech2):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def call(self, inputs, training=False):
        input_ids, speaker_ids, speed_ratios, f0_ratios, energy_ratios = inputs
        return super()._inference(
            input_ids, speaker_ids, speed_ratios, f0_ratios, energy_ratios
        )

    def _build(self):
        input_ids = tf.convert_to_tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], tf.int32)
        speaker_ids = tf.convert_to_tensor([0], tf.int32)
        speed_ratios = tf.convert_to_tensor([1.0], tf.float32)
        f0_ratios = tf.convert_to_tensor([1.0], tf.float32)
        energy_ratios = tf.convert_to_tensor([1.0], tf.float32)
        self([input_ids, speaker_ids, speed_ratios, f0_ratios, energy_ratios])
