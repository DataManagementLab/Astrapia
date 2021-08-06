from xaibenchmark.samplers.base_sampler import Sampler

class SPLimeSampler(Sampler):

    def sample(self, data, n):
        """
        Sample n elements from data.
        """
        raise NotImplementedError("SPLimeSampler.sample is not implemented.")