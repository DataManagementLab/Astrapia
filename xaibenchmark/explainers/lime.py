import numpy as np
from xaibenchmark import preprocessing
from xaibenchmark import Explainer
import xaibenchmark as xb

import lime
import lime.lime_tabular


class LimeExplainer(Explainer):

    def __init__(self, data, predict_fn, categorical_features=None, discretize_continuous=True):

        self.categorical_features = data.categorical_features
        self.data_keys = data.data.keys()

        train, dev, test = preprocessing.lime_preprocess_datasets(
            [data.data, data.data_dev, data.data_test], self.categorical_features, self.data_keys)

        self.explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=train.keys(),
                                                                class_names=data.target_names, categorical_features=None,
                                                                discretize_continuous=discretize_continuous)
        self.train = train
        self.dev = dev
        self.test = test
        self.predict_fn = predict_fn
        self.kernel_width = np.sqrt(train.shape[1]) * .75

    def explain_instance(self, instance, num_features=10):
        instance = preprocessing.lime_preprocess_dataset(instance, self.categorical_features, self.data_keys).iloc[0]
        self.explanation = self.explainer.explain_instance(instance, self.predict_fn.predict_proba,
                                                           num_features=num_features)
        self.instance = instance
        self.weighted_instances = self.get_weighted_instances()

        return self.explanation

    @xb.metric
    def area(self):
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

        ml_preds = self.predict_fn.predict_proba(self.train)[:, 1]
        ml_preds = ml_preds > 0.5
        exp_preds = [self.predict_instance_surrogate(instance) for instance, _ in self.weighted_instances]
        exp_preds = np.array(exp_preds) > 0.5
        return (ml_preds == exp_preds).sum() / len(exp_preds)

    @xb.metric
    def balance(self):
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
