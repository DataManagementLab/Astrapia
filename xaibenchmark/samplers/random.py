from xaibenchmark.samplers.base_sampler import Sampler
import random

class RandomSampler(Sampler):

    def sample(self, data, n):
        """
        Sample n elements from data.
        """
        indexes = random.sample(range(data.shape[0]), n)
        return data.iloc[list(indexes)]