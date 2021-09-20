from collections import defaultdict
from astrapia.samplers import base_sampler, random, splime
import astrapia as xb
from datetime import datetime
import json
from tqdm import tqdm
import pandas as pd


class ExplainerComparator:
    """
    A comparator that allows the user to add explainers, let them explain instances and store metrics from the
    different explainers
    """

    def __init__(self):
        # Dictionary with key: name of explainer, value: explainer as object
        self.explainers = {}

        # Dictionary with key: name of explainer, value: dictionary with key: name of property, value: property
        self.properties = {}

        # Dictionary with key: name of explainer, value: dictionary with key: name of metric, value: average value
        self.averaged_metrics = {}

        # Dictionary with key: name of explainer, value: dictionary with key: index of explanation,
        # value: explanation as object
        self.explanations = {}

        # Dictionary with key: name of explainer, value: dictionary with key: index of explanation,
        # value: dictionary with key: name of metric, value: metric value
        self.metrics = {}

        # instances that are used to create explanations
        self.instances = None

        # timestamp of the creation of metrics
        self.timestamp = ''

    def add_explainer(self, explainer: xb.Explainer, name: str):
        """
        Add an instantiated explainer to the comparator. Use the name attribute for uniquely identifying different
        explainers (E.g. between 'Anchors acc>95%' and 'Anchors acc>85%')

        :param explainer: explainer object
        :param name: unique name for identifying the explainer
        """

        explainer_properties = {}
        for (prop, value) in explainer.report(tag='prop', inferred_metrics=False):
            explainer_properties[prop] = value

        self.explainers[name] = explainer
        self.properties[name] = explainer_properties

    def explain_instances(self, instances: pd.DataFrame, inferred_metrics=False):
        """
        Create explanations for all combinations of provided explainers and instances, then save metrics

        :param inferred_metrics: Check whether you want to include inferred metrics in the report
        :param instances: instances to be used to create explanations as pandas dataframe
        """

        # Reset aggregation attributes
        self.averaged_metrics = {}
        self.explanations = {}
        self.metrics = {}
        self.instances = instances

        # Initialize tqdm progress bar
        with tqdm(total=len(self.explainers.keys()) * len(instances)) as pbar:

            # Iterate over all explainers
            for name, explainer in self.explainers.items():
                explainer_average_metrics = defaultdict(float)
                aggregated_explainer_metrics = {}
                aggregated_explanations = {}

                # iterate over all instances
                for index in range(instances.shape[0]):
                    explanation = explainer.explain_instance(instances.iloc[[index]])

                    explanation_metrics = {}
                    for (metric, value) in explainer.report(tag='metric', inferred_metrics=inferred_metrics):
                        # add up metrics in order to compute average values later
                        explainer_average_metrics[metric] += value
                        # save metrics of the explanation separately
                        explanation_metrics[metric] = value

                    aggregated_explainer_metrics[str(index)] = explanation_metrics
                    aggregated_explanations[str(index)] = explanation

                    # update progress bar
                    pbar.update(1)

                # compute average metrics by dividing added metrics by the amount of provided instances
                explainer_average_metrics = {metric: value / instances.shape[0]
                                             for metric, value in explainer_average_metrics.items()}

                self.averaged_metrics[name] = explainer_average_metrics
                self.metrics[name] = aggregated_explainer_metrics
                self.explanations[name] = aggregated_explanations
            self.timestamp = str(datetime.now())

    def explain_representative(self, data: xb.Dataset, sampler: str = 'splime', count: int = 10, pred_fn=None,
                               return_samples: bool = False, inferred_metrics=False, **kwargs):
        """
        Create a representative explanation for the given data

        :param inferred_metrics: Check whether you want to include inferred metrics in the report
        :param return_samples: Check whether sampled elements should be returned
        :param pred_fn: Provide optional prediction function for sampling
        :param data: pandas dataframe with the data to be explained
        :param sampler: sampler to be used to create representative explanation
        :param count: amount of representative samples to be created
        """

        # Map sampler names to objects
        # Add new samplers here
        sampler_map = {
            'random': random.RandomSampler,
            'splime': splime.SPLimeSampler,
        }

        if issubclass(type(sampler), base_sampler.Sampler):
            # sampler is an object
            sampler = sampler()
        elif sampler in sampler_map:
            # sampler is a valid string
            sampler = sampler_map[sampler]()
        else:
            # sampler is unknown
            raise NameError('Invalid sampler \'' + sampler + '\'')

        # sampler is now a Sampler object
        # sample instances and explain them
        instances = sampler.sample(data, count, pred_fn, **kwargs)

        # relay the samples instances to the regular explain_instances function
        self.explain_instances(instances, inferred_metrics=inferred_metrics)

        if return_samples:
            return instances

    def store_metrics(self, filename: str = 'metrics'):
        """
        Store metric data in a json file

        :param filename: name of the file
        :return: metric data as dictionary
        """
        data = self.get_metric_data()
        with open(filename + '.json', 'w') as outfile:
            json.dump(data, outfile)
        return data

    def get_metric_data(self):
        """
        Get metric data including the timestamp of the creation of metrics

        :return: metric data as dictionary
        """
        return {'timestamp': self.timestamp, 'explainers': list(self.explainers.keys()), 'properties': self.properties,
                'averaged_metrics': self.averaged_metrics, 'separate_metrics': self.metrics}
