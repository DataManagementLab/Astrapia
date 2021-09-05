import xaibenchmark as xb

class Sampler:
    
    def sample(self, data: xb.Dataset, count: int, pred_fn):
        """
        Sample n elements from data.
        """
        raise NotImplementedError("Sampler.sample is not implemented.")