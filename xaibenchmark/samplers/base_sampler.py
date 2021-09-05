import xaibenchmark as xb

class Sampler:
    
    def sample(self, data: xb.Dataset, count: int, **kwargs):
        """
        Sample n elements from data.

        :param data: The dataset to sample from.
        :param count: The number of elements to sample.
        :return: A list of elements sampled from data.
        """
        raise NotImplementedError("Sampler.sample is not implemented.")