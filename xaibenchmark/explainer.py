from __future__ import print_function
import xaibenchmark as xb
import pandas as pd


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
        """
        Extend this class and override this method to define your explainer
        """
        raise NotImplementedError

    def transform_dataset(self, data: pd.DataFrame, meta: xb.Dataset) -> any:
        """
        Transforms the given dataset into a dataset readable by the explainer

        :param data: the dataset to transform
        :param meta: Dataset object containing metadata
        :return: the transformed dataset            
        """
        return data.copy()

    def inverse_transform_dataset(self, data: any, meta: xb.Dataset) -> pd.DataFrame:
        """
        Inverse transforms the given dataset into a general format to be used by a model or a sampler

        :param data: the dataset to inverse transform
        :param meta: Dataset object containing metadata
        :return: the inverse transformed dataset
        """
        return data.copy()
        
    def metrics(self) -> list:
        """
        Returns a list of metrics that are available for this explainer

        :return: a list of metrics
        """
        return [x for x in dir(self) if getattr(getattr(self, x), 'tag', None) == 'metric']
    
    def infer_metrics(self, printing:bool=True) -> None:
        """
        Infer missing metrics for this explainer.
        Metrics are inferred using the transfer graph.

        To add metric transfers, look into xaibenchmark.transfer

        :param printing: whether to print the metrics or not
        """
        xb.transfer.use_transfer(self)
        if printing:
            print('inferred metrics:', {x for x in dir(self) if getattr(getattr(self, x), 'tag', None) in ['metric', 'utility']})
        
    def report(self) -> dict:
        """
        Compute and print metrics for this explainer

        :return: a dictionary of metrics
        """
        all_mu_identifier_references = {(x, getattr(self, x)) for x in dir(self) if getattr(getattr(self, x), 'tag', None) == 'metric'}
        implemented_mu_values = {(x, f()) for (x, f) in all_mu_identifier_references}
        return implemented_mu_values





