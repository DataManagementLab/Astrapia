from xaibenchmark.samplers.base_sampler import Sampler
import xaibenchmark as xb
import random

class RandomSampler(Sampler):

    def sample(self, data: xb.Dataset, n: int, **kwargs):
        """
        Sample n random elements from the dataset.

        :param data: The dataset to sample from.
        :param n: The number of elements to sample.
        :return: A pandas DataFrame of n elements.
        """

        indexes = random.sample(range(data.data.shape[0]), n)
        return data.data.iloc[list(indexes)]