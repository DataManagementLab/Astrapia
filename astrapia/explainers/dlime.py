import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors

import astrapia as xb
from astrapia.explainer import Explainer
from astrapia.explainers.DLime.explainer_tabular import LimeTabularExplainer as DLimeTabularExplainer


class DLimeExplainer(Explainer):
    """
    Implementation of the DLime Explainer onto the base Explainer class
    """

    def __init__(self, data, predict_fn, discretize_continuous=True):
        """
        Initializes a DLime explainer

        :param data: data that is supposed to be explained
        :param predict_fn: classification model that is supposed to be explained
        :param discretize_continuous: should continuous values be separated into discrete categories
        """
        self.categorical_features = data.categorical_features
        self.data_keys = data.data.keys()
        self.data = data

        self.train = self.transform_dataset(data.data, data)
        self.dev = self.transform_dataset(data.data_dev, data)
        self.test = self.transform_dataset(data.data_test, data)

        self.explainer = DLimeTabularExplainer(self.train,
                                               mode="classification",
                                               feature_names=self.train.keys(),
                                               class_names=data.target_names,
                                               categorical_features=None,
                                               discretize_continuous=discretize_continuous)

        clustering = AgglomerativeClustering().fit(self.train)
        self.clustered_data = np.column_stack([self.train, clustering.labels_])

        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.train)
        distances, self.indices = nbrs.kneighbors(self.test)
        self.clabel = clustering.labels_

        self.predict = predict_fn
        self.kernel_width = np.sqrt(self.train.shape[1]) * .75

    def transform_dataset(self, data: pd.DataFrame, meta: xb.Dataset) -> any:
        """
        Returns the onehot encoded dataset

        :param data: given dataset
        :param meta: metadata with categorical information
        :return: One-hot encoded DataFrame
        """
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
            max_indices = np.argmax(
                data[[feature + '_' + str(l) for l in meta.categorical_features[feature]]].to_numpy(), axis=1)
            df[feature] = pd.Series([meta.categorical_features[feature][f] for f in max_indices], index=data.index)

        continuous = list(meta.feature_names - meta.categorical_features.keys())
        df[continuous] = data[continuous]
        return df[meta.data.keys()]

    def explain_instance(self, instance, num_features=10):
        """
        Creates a dlime explanation based on a given instance

        :param instance: instance as dataframe
        :param num_features: amount of features in the dataset
        :return: the explanation
        """

        self.instance = self.transform_dataset(instance, self.data).iloc[0]

        p_label = self.clabel[self.indices[0]]
        N = self.clustered_data[self.clustered_data[:, 30] == p_label]
        subset = np.delete(N, 30, axis=1)

        self.explanation = self.explainer.explain_instance_hclust(self.instance,
                                                                 lambda x: self.predict(self.inverse_transform_dataset(
                                                                     pd.DataFrame(x, columns=self.train.keys()), self.data)),
                                                                 num_features=num_features,
                                                                 model_regressor=LinearRegression(),
                                                                 clustered_data=subset,
                                                                 regressor='linear',
                                                                 labels=(0, 1))
        self.weighted_instances = self.get_weighted_instances()

        return self.explanation

    @xb.utility
    def predict_instance_surrogate(self, instance):
        """
        Helper function for accessing the predictions of lime's surrogate model

        :param instance: instance whose prediction should be provided
        :return: label prediction of given instance
        """
        return np.clip(self.explanation.intercept[1] + sum(weight * ((instance - self.explainer.scaler.mean_) /
                                                                     self.explainer.scaler.scale_)[idx]
                                                           for idx, weight in self.explanation.local_exp[1]), 0, 1)

    @xb.prop
    def shape(self):
        return 'Exponential kernel'

    @xb.prop
    def name(self):
        return 'DLime'

    @xb.prop
    def instancing(self):
        return 'weighted'

    @xb.metric
    def area_absolute(self):
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
        return sum([weight for _, weight in self.weighted_instances]) / len(self.weighted_instances)

    @xb.metric
    def coverage_absolute(self):
        """
        Number of instances within the neighbourhood.
        """
        return sum(weight for _, weight in self.weighted_instances)

    @xb.metric
    def distance_furthest(self):
        """
        Highest distance between any two instances that are in the neighborhood of the explanation

        :return: distance value
        """
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

        :return: the accuracy value
        """

        ml_preds = self.predict(self.inverse_transform_dataset(self.train, self.data))
        ml_preds = ml_preds[:, 1] > 0.5
        exp_preds = [self.predict_instance_surrogate(instance) for instance, _ in self.weighted_instances]
        exp_preds = np.array(exp_preds) > 0.5
        weights = np.array([weight for _, weight in self.weighted_instances])
        return ((ml_preds == exp_preds) * weights).sum() / sum(weights)


    @xb.metric
    def balance_explanation(self):
        """
        Relative amount of data elements in the explanation neighborhood that had an assigned label value of 1
        (by the explanation)

        :return: the balance value
        """

        if hasattr(self, 'explanation'):
            exp_preds = [self.predict_instance_surrogate(instance) for instance, _ in self.weighted_instances]
            exp_preds = np.array(exp_preds) > 0.5

            weights = np.array([weight for _, weight in self.weighted_instances])
            return (exp_preds * weights).sum() / sum(weights)

    @xb.metric
    def balance_model(self):
        """
        Relative amount of data elements in the neighborhood of the explanation that had a label value of 1 assigned
        by the classification model

        :return: the balance value
        """
        if hasattr(self, 'explanation'):
            ml_preds = self.predict(self.inverse_transform_dataset(self.train, self.data))
            ml_preds = ml_preds[:, 1] > 0.5

            weights = np.array([weight for _, weight in self.weighted_instances])
            return (ml_preds * weights).sum() / sum(weights)

    @xb.metric
    def balance_data(self):
        """
        Relative amount of data elements in the neighborhood of the explanation with a label value of 1

        :return: the balance value
        """
        if hasattr(self, 'explanation'):
            weights = np.array([weight for _, weight in self.weighted_instances])
            return sum((self.data.target.to_numpy().reshape((-1,)) == self.data.target_names[1]) * weights) / sum(
                weights)

    @xb.metric
    def accuracy_global(self):
        """
        Proportion of instances in the full data space that shares the same output label by the
        explainer and the ML model

        :return: the accuracy value
        """

        ml_preds = self.predict(self.inverse_transform_dataset(self.train, self.data))
        ml_preds = ml_preds[:, 1] > 0.5
        exp_preds = [self.predict_instance_surrogate(instance) for instance, _ in self.weighted_instances]
        exp_preds = np.array(exp_preds) > 0.5
        return (ml_preds == exp_preds).sum() / len(ml_preds)

    @xb.utility
    def distance(self, x, y):
        """
        calculates the euclidean distance between two data points

        :param x: first point
        :param y: second point
        :return: distance
        """
        return np.linalg.norm(x - y)

    @xb.utility
    def get_weighted_instances(self):
        """
        returns instances associated with their weight concerning the explanation

        :return: List of tuples with instance and its weight
        """
        if hasattr(self, 'explanation'):
            kernel_width = np.sqrt(self.train.shape[1]) * .75

            def kernel(distance):
                return np.sqrt(np.exp(-distance ** 2 / kernel_width ** 2))

            return [(instance, kernel(self.distance(self.instance, instance)))
                    for instance in self.train.to_numpy()]
        return []

    @xb.utility
    def get_explained_instance(self):
        """
        Returns instance that was explained

        :return: instance
        """
        return self.instance
