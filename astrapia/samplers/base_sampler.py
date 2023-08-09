import astrapia as xb


class Sampler:
    """
    The Sampler class provides a simple interface for sampling representative instances. This class should not be used as-is but rather extended.
    """

    def sample(self, data: xb.Dataset, count: int, *args, **kwargs):
        """
        Sample n elements from data.

        :param data: The dataset to sample from.
        :param count: The number of elements to sample.
        :param kwargs: Additional arguments possibly required for more sophisticated samplers.
        :return: A pandas DataFrame of representative samples.
        """
        raise NotImplementedError("Sampler.sample is not implemented.")
