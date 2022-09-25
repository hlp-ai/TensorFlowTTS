"""Dataset modules."""
import os

import numpy as np
import tensorflow as tf

from tensorflow_tts.datasets.abstract_dataset import AbstractDataset
from tensorflow_tts.utils import find_files


class CharactorDataset(AbstractDataset):
    """Tensorflow Charactor dataset."""

    def __init__(
        self, root_dir, charactor_query="*-ids.npy", charactor_load_fn=np.load,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            charactor_query (str): Query to find charactor files in root_dir.
            charactor_load_fn (func): Function to load charactor file.
            return_utt_id (bool): Whether to return the utterance id with arrays.

        """
        # find all of charactor and mel files.
        charactor_files = sorted(find_files(root_dir, charactor_query))

        # assert the number of files
        assert (
            len(charactor_files) != 0
        ), f"Not found any char or duration files in ${root_dir}."
        if ".npy" in charactor_query:
            suffix = charactor_query[1:]
            utt_ids = [os.path.basename(f).replace(suffix, "") for f in charactor_files]

        # set global params
        self.utt_ids = utt_ids
        self.charactor_files = charactor_files
        self.charactor_load_fn = charactor_load_fn

    def get_args(self):
        return [self.utt_ids]

    def generator(self, utt_ids):
        for i, utt_id in enumerate(utt_ids):
            charactor_file = self.charactor_files[i]
            charactor = self.charactor_load_fn(charactor_file)

            items = {"utt_ids": utt_id, "input_ids": charactor}

            yield items

    def create(
        self,
        allow_cache=False,
        batch_size=1,
        is_shuffle=False,
        map_fn=None,
        reshuffle_each_iteration=True,
    ):
        """Create tf.dataset function."""
        output_types = self.get_output_dtypes()
        datasets = tf.data.Dataset.from_generator(
            self.generator, output_types=output_types, args=(self.get_args())
        )

        if allow_cache:
            datasets = datasets.cache()

        if is_shuffle:
            datasets = datasets.shuffle(
                self.get_len_dataset(),
                reshuffle_each_iteration=reshuffle_each_iteration,
            )

        # define padded shapes
        padded_shapes = {"utt_ids": [], "input_ids": [None]}

        datasets = datasets.padded_batch(
            batch_size, padded_shapes=padded_shapes, drop_remainder=True
        )
        datasets = datasets.prefetch(tf.data.experimental.AUTOTUNE)
        return datasets

    def get_output_dtypes(self):
        output_types = {"utt_ids": tf.string, "input_ids": tf.int32}
        return output_types

    def get_len_dataset(self):
        return len(self.utt_ids)

    def __name__(self):
        return "CharactorDataset"
