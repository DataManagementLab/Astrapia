from __future__ import print_function
import astrapia as xb
import pandas as pd


class Explainer:
    """
    The Explainer class wraps an explainer and provides a unified interface for it. 
    Initialization depends on the specific explainer. 
    This class should *not* be used as is but rather extended.
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

    def explain_instance(self, instance: pd.DataFrame):
        """
        Used to generate an explanation of a single instance. 
        Should be overriden in every subclass of Explainer.

        :param instance: the instance to be explained
        """
        raise NotImplementedError
        
    def metrics(self) -> list:
        """
        Returns a list of metrics that are available for this explainer

        :return: a list of metric references
        """
        return [x for x in dir(self) if getattr(getattr(self, x), 'tag', None) == 'metric']

    def props(self) -> list:
        """
        Returns a list of properties that are available for this explainer

        :return: a list of property references
        """
        return [x for x in dir(self) if getattr(getattr(self, x), 'tag', None) == 'prop']
    
    def infer_metrics(self, printing: bool = True) -> None:
        """
        Infer missing metrics for this explainer.
        Metrics are inferred using the transfer graph.

        To add metric transfers, look into astrapia.transfer

        :param printing: whether to print the metrics or not
        """
        xb.transfer.use_transfer(self)
        if printing:
            print('inferred metrics:', {x for x in dir(self) if getattr(getattr(self, x), 'tag', None) in ['metric', 'utility']})
        
    def report(self, tag=None, inferred_metrics=True) -> dict:
        """
        Compute metrics and properties for this explainer.
        If a tag is supplied, only the respective type of attribute is returned (metrics or properties)

        :param inferred_metrics:
        :param tag: *None* or 'prop' or 'metric'
        :return: a dictionary of metrics
        """
        if inferred_metrics:
            self.infer_metrics()

        if tag is None:
            all_mu_identifier_references = {(x, getattr(self, x)) for x in dir(self) if getattr(getattr(self, x), 'tag', None) in ['metric', 'prop']}
        elif tag == 'metric':
            all_mu_identifier_references = {(x, getattr(self, x)) for x in dir(self) if getattr(getattr(self, x), 'tag', None) == 'metric'}
        elif tag == 'prop':
            all_mu_identifier_references = {(x, getattr(self, x)) for x in dir(self) if getattr(getattr(self, x), 'tag', None) == 'prop'}
        else:
            raise ValueError(f'Tag should be either "metric" or "prop", not ${tag}')
            
        implemented_mu_values = {(x, f()) for (x, f) in all_mu_identifier_references}
        return implemented_mu_values
