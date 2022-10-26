"""Abstract Dataset modules."""
import abc


class AbstractDataset(metaclass=abc.ABCMeta):
    """Abstract Dataset module for Dataset Loader."""

    @abc.abstractmethod
    def get_args(self):
        """Return args for generator function."""
        pass

    @abc.abstractmethod
    def generator(self):
        """Generator function, should have args from get_args function."""
        pass

    @abc.abstractmethod
    def get_output_dtypes(self):
        """Return output dtypes for each element from generator."""
        pass

    @abc.abstractmethod
    def get_len_dataset(self):
        """Return number of samples on dataset."""
        pass

    def create(
        self,
        allow_cache=False,
        batch_size=1,
        is_shuffle=False,
        map_fn=None,
        reshuffle_each_iteration=True,
    ):
        pass
