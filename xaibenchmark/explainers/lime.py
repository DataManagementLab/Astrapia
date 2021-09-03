import numpy as np
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from xaibenchmark import preprocessing
from xaibenchmark import Explainer
import xaibenchmark as xb
import pandas as pd

from functools import reduce

import lime
import lime.lime_tabular


class LimeExplainer(Explainer):

    def __init__(self, data, predict_fn, discretize_continuous=True):

        self.categorical_features = data.categorical_features
        self.data_keys = data.data.keys()
        self.data = data

        self.train = self.transform_dataset(data.data, data)
        self.dev = self.transform_dataset(data.data_dev, data)
        self.test = self.transform_dataset(data.data_test, data)

        self.explainer = lime.lime_tabular.LimeTabularExplainer(self.train, feature_names=self.train.keys(),
                                                                class_names=data.target_names, categorical_features=None,
                                                                discretize_continuous=discretize_continuous)

        self.predict = lambda x: predict_fn(self.inverse_transform_dataset(x, data))
        self.kernel_width = np.sqrt(self.train.shape[1]) * .75

    def transform_dataset(self, data: pd.DataFrame, meta: xb.Dataset) -> any:

        return xb.utils.onehot_encode(data, meta)

    def inverse_transform_dataset(self, data: pd.DataFrame, meta: xb.Dataset):
        """
        Inverse transform an explainer-specific dataset into the general Astrapia Dataset format

        :param data: pandas dataframe holding data in the shape LIME needs it
        :param meta: Astrapia Dataset object holding meta information that does not depend on data instances
        :returns: pandas dataframe in Astrapia Dataset format
        """
        df = pd.DataFrame(index=data.index)
        for feature in meta.categorical_features:

            max_indices = np.argmax(data[[feature+'_'+str(l) for l in meta.categorical_features[feature]]].to_numpy(), axis=1)
            df[feature] = pd.Series([meta.categorical_features[feature][f] for f in max_indices], index=data.index)
            
        continuous = list(meta.feature_names - meta.categorical_features.keys())
        df[continuous] = data[continuous]
        return df[meta.data.keys()]

    def explain_instance(self, instance, num_features=10):
        instance = self.transform_dataset(instance, self.data).iloc[0]
        self.explanation = self.explainer.explain_instance(instance, lambda x: self.predict(pd.DataFrame(x, columns=self.train.keys()), self.data),
                                                           num_features=num_features)
        self.instance = instance
        self.weighted_instances = self.get_weighted_instances()

        return self.explanation

    @xb.prop
    def shape(self):
        return 'Exponential kernel'

    @xb.prop
    def name(self):
        return 'Lime'

    @xb.metric
    def absolute_area(self):
        """
        Area that is covered by the kernel in high dimension of the feature space.
        """
        kernel_width = np.sqrt(self.train.shape[1]) * .75
        kernel_dimension = self.train.shape[1]
        return (kernel_width * np.sqrt(2 * np.pi)) ** kernel_dimension

    @xb.metric
    def coverage(self):
        """
        Proportion of instances covered in the area
        """
        weighted_instances = self.weighted_instances
        return sum([weight for _, weight in self.weighted_instances]) / len(self.weighted_instances)

    @xb.metric
    def furthest_distance(self):
        kernel_width = np.sqrt(self.train.shape[1]) * .75

        def kernel(distance):
            return np.sqrt(np.exp(-distance ** 2 / kernel_width ** 2))

        training_instances = self.train.to_numpy()
        distance_instances = (self.distance(self.instance, instance) for instance in training_instances)
        weighted_distances = (distance * kernel(distance) for distance in distance_instances)
        return sum(weighted_distances)

    @xb.metric
    def accuracy(self):
        """
        Proportion of instances in the explanation neighborhood that shares the same output label by the
        explainer and the ML model
        """

        ml_preds = self.predict(self.train)
        ml_preds = ml_preds[:,1] > 0.5
        exp_preds = [self.predict_instance_surrogate(instance) for instance,_ in self.weighted_instances]
        exp_preds = np.array(exp_preds) > 0.5
        weights = np.array([weight for _, weight in self.weighted_instances])
        return ((ml_preds == exp_preds) * weights).sum() / sum(weights)

    @xb.metric
    def balance_explanation(self):
        """
        Proportion of instances in the explanation neighborhood that has been assigned label 1 by the
        explanation model
        """
        exp_preds = [self.predict_instance_surrogate(instance) for instance, _ in self.weighted_instances]
        exp_preds = np.array(exp_preds) > 0.5
        return exp_preds.sum() / len(exp_preds)

    @xb.utility
    def distance(self, x, y):
        return np.linalg.norm(x - y)

    @xb.utility
    def get_weighted_instances(self):
        if hasattr(self, 'explanation'):
            kernel_width = np.sqrt(self.train.shape[1]) * .75

            def kernel(distance):
                return np.sqrt(np.exp(-distance ** 2 / kernel_width ** 2))

            return [(instance, kernel(self.distance(self.instance, instance)))
                    for instance in self.train.to_numpy()]
        return []

    @xb.utility
    def get_explained_instance(self):
        return self.instance

    @xb.utility
    def get_training_data(self):
        return self.train

    @xb.utility
    def predict_instance_surrogate(self, instance):
        return np.clip(self.explanation.intercept[1] + sum(weight * ((instance - self.explainer.scaler.mean_) /
                                                                     self.explainer.scaler.scale_)[idx]
                                                           for idx, weight in self.explanation.local_exp[1]), 0, 1)
