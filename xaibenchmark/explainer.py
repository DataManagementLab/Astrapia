from __future__ import print_function
import xaibenchmark as xb


class Explainer:
    """Base class for wrapping and comparing Explainers
    
    Add metrics using the @metric decorator.
    Add utility functions using the @utility decorator.

    When using the following predefined metrics and utilities, the library can infer other metrics by calling 
        your_explainer.infer_metrics()

    Metrics:
    - coverage(self)
    - 

    Utilities:
    - distance(self, x, y)
    - get_neighborhood_instances(self)
    -
    """
    
    def __init__(self):
        raise NotImplementedError
        
    def metrics(self):
        return [x for x in dir(self) if getattr(getattr(self, x), 'tag', None) == 'metric']
    
    def infer_metrics(self, printing=True):
        xb.transfer.use_transfer(self)
        if printing:
            print('inferred metrics:', {x for x in dir(self) if getattr(getattr(self, x), 'tag', None) in ['metric', 'utility']})
        
    def report(self):
        all_mu_identifier_references = {(x, getattr(self, x)) for x in dir(self) if getattr(getattr(self, x), 'tag', None) == 'metric'}
        implemented_mu_values = {(x, f()) for (x, f) in all_mu_identifier_references}
        return implemented_mu_values





