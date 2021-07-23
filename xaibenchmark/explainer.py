from __future__ import print_function
import numpy as np
import xaibenchmark as xb
from xaibenchmark import load_adult
from xaibenchmark import customAnchorsPreprocessing
from anchor import anchor_tabular
from anchor import utils
import lime
import lime.lime_tabular
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


class AnchorsExplainer(Explainer):
    """
    implementation of the Explainer "Anchors" onto the base explainer class
    """

    def __init__(self, predictor, dataset_folder):

        dataset = utils.load_dataset('adult', balance=True, dataset_folder=dataset_folder, discretize=True)
        self.rawdata = load_adult.load_csv_data('adult', root_path='../data')

        self.explainer = anchor_tabular.AnchorTabularExplainer(
            dataset.class_names,
            dataset.feature_names,
            dataset.train,
            dataset.categorical_names)
        self.dataset = dataset
        self.predictor = predictor

    def get_subset(self, subset_name):
        """
        Returns one of the 3 subsets given a name
        :param subset_name: either train, dev or test
        :return: data subset as ndarray
        """
        if subset_name == "train":
            return self.dataset.train, self.dataset.labels_train
        elif subset_name == "dev":
            return self.dataset.validation, self.dataset.labels_validation
        elif subset_name == "test":
            return self.dataset.test, self.dataset.labels_test
        else:
            raise NameError("This subset name is not one of train, dev, test.")

    def explain_instance(self, instance, instance_set="train", threshold=0.70):
        """
        Creates an Anchor explanation based on a given instance
        :param instance: "Anchor" for explanation
        :param instance_set: textual information about subset for metric information
        :param threshold: Worst possible precision for the explanation
        :return: the explanation
        """

        instance = self.preprocessInstance(instance)

        self.explanation = self.explainer.explain_instance(instance, self.predictor.predict, threshold=threshold)
        self.instance = instance
        self.instance_set, self.instance_label_set = self.get_subset(instance_set)
        return self.explanation

    @xb.metric
    def coverage(self):
        """
        The relative amount of data elements that are in the area of the explanation
        :return: the coverage value
        """
        if hasattr(self, 'explanation'):
            return self.explanation.coverage()
        return np.nan

    @xb.metric
    def precision(self):
        """
        The ML-accuracy of the explanation when applied to the whole dataset (not just the area of the explanation)
        :return: the precision value
        """
        if hasattr(self, 'explanation'):
            return self.explanation.precision()
        return np.nan

    @xb.metric
    def balance_explanation(self):
        """
        New implementation of balance:
        Relative amount of data elements in the explanation neighborhood that had an assigned label value of 1
        (by the explanation)
        :return: the balance value
        """
        # balance is always 0 or 1 because Anchors creates a neighborhood where all elements are supposed to have
        # the same label as the one that was used to instantiate the explanation
        return self.explanation.exp_map["prediction"]

    @xb.metric
    def balance_data(self, dataset=None, labelset=None):
        """
        Relative amount of data elements in the given set or set of the original instance with a label value of 1
        :return: the balance value
        """
        if hasattr(self, 'explanation'):
            if (dataset is None) ^ (labelset is None):
                raise NameError("Either declare dataset and labelset or none of them")
            elif dataset is None:
                dataset = self.instance_set
                labelset = self.instance_label_set
            return np.mean(labelset[self.get_fit_anchor(dataset)])

    @xb.metric
    def balance_model(self, dataset=None):
        """
        Relative amount of data elements in the neighborhood of the explanation or the given set
        with an assigned label (by the ML model) value of 1
        """
        if hasattr(self, 'explanation'):
            if dataset is None:
                dataset = self.instance_set
            relevant_examples = dataset[self.get_fit_anchor(dataset)]
            if len(relevant_examples) > 0:
                return np.mean(self.predictor.predict(relevant_examples))

    @xb.metric
    def area(self):
        """
        Relative amount of feature space over all features n that is specified by the explanation.
        area = Product[i=1->n] fi, f: 1 if feature is not in explanation, else 1/m, m: deminsionality of feature
        :return: the area value
        """
        if hasattr(self, 'explanation'):
            array = np.amax(self.dataset.train, axis=0)[self.explanation.features()]
            array = array + 1
            # 1/4*1*1*1*1/10 = 2.5%
            # optionally with n-th root. n=amount of features or dimension of features?
            # print(np.power(np.prod(1 / array), 1/len(array)), np.power(np.prod(1 / array), 1/np.sum(array)))
            return np.prod(1 / array)
        return np.nan

    @xb.metric
    def accuracy(self, dataset=None):
        """
        Relative amount of data elements in explanation neighborhood or given dataset that have the same explanation
        label as the label assigned by the ML model
        :return: the accuracy value
        """
        if hasattr(self, 'explanation'):
            if dataset is None:
                dataset = self.instance_set
            explanation_label = self.explanation.exp_map["prediction"]
            relevant_examples = dataset[self.get_fit_anchor(dataset)]
            if len(relevant_examples) > 0:
                ml_pred = self.predictor.predict(relevant_examples)
                return np.count_nonzero(ml_pred == explanation_label) / len(relevant_examples)

    @xb.utility
    def get_neighborhood_instances(self):
        """
        Receive all data elements in the given subset that belong to the neighborhood of the explanation
        :return: ndarray of elements
        """
        if hasattr(self, 'explanation'):
            fit_anchor = self.get_fit_anchor(self.instance_set)
            return self.instance_set[fit_anchor]
        return []

    @xb.utility
    def get_fit_anchor(self, dataset):
        return np.where(np.all(dataset[:, self.explanation.features()] ==
                                         self.instance[self.explanation.features()], axis=1))[0]
    @xb.utility
    def get_explained_instance(self):
        return self.instance

    @xb.utility
    def distance(self, x, y):
        return np.linalg.norm(x-y)

    def preprocessInstance(self, instance):
        return customAnchorsPreprocessing.load_dataset(self.rawdata.data.append(instance, ignore_index=True).to_numpy())


class LimeExplainer(Explainer):

    def __init__(self, data, predict_fn, categorical_features=None, discretize_continuous=True):

        self.categorical_features = data.categorical_features
        self.data_keys = data.data.keys()

        def preprocess(*data_df):
            return [self.process_single(df) for df in data_df]

        train, dev, test = preprocess(data.data, data.data_dev, data.data_test)
        labels_train, labels_dev, labels_test = data.target, data.target_dev, data.target_test

        self.explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=train.keys(),
                                                                class_names=data.target_names, categorical_features=None,
                                                                discretize_continuous=discretize_continuous)
        self.train = train
        self.dev = dev
        self.test = test
        self.predict_fn = predict_fn
        self.kernel_width = np.sqrt(train.shape[1]) * .75

    def process_single(self, df):
        cat_df = pd.get_dummies(df, columns=self.categorical_features.keys())
        missing_cols = {cat + '_' + str(attr) for cat in self.categorical_features \
                        for attr in self.categorical_features[cat]} - set(cat_df.columns)
        for c in missing_cols:
            cat_df[c] = 0

        cont_idx = list(set(self.data_keys) - set(self.categorical_features.keys()))
        cat_idx = [cat + '_' + str(attr) for cat in self.categorical_features \
                   for attr in self.categorical_features[cat]]
        idx = cont_idx + cat_idx
        return cat_df[idx]

    def explain_instance(self, instance, num_features=10):
        instance = self.process_single(instance).iloc[0]
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

            return [(instance, kernel(self.distance(self.instance, instance))) \
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
        return np.clip(self.explanation.intercept[1] + sum(weight * \
                                                           ((
                                                                        instance - self.explainer.scaler.mean_) / self.explainer.scaler.scale_)[
                                                               idx] \
                                                           for idx, weight in self.explanation.local_exp[1]), 0, 1)
