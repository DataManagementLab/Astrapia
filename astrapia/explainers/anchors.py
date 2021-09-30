import numpy as np
from anchor import anchor_tabular
from astrapia import Explainer
import astrapia as xb
import pandas as pd


class AnchorsExplainer(Explainer):
    """
    Implementation of the Anchors Explainer onto the base Explainer class
    """

    def __init__(self, data, predict_fn, min_precision=0.9):
        """
        Initializes an Anchors explainer

        :param data: data to be explained
        :param predict_fn: prediction function
        :param min_precision: minimum precision of the anchor
        """

        self.anchors_dataset = self.transform_dataset(data.data, data)
        self.min_precision = min_precision

        self.explainer = anchor_tabular.AnchorTabularExplainer(
            self.anchors_dataset['class_names'],
            self.anchors_dataset['feature_names'],
            self.anchors_dataset['data'],
            self.anchors_dataset['categorical_names'])
        self.meta = data
        
        def transformed_predict(data):
            return predict_fn(self.inverse_transform_dataset({'data': data}, self.meta))[:,1] > 0.5
        self.predictor = transformed_predict

    def transform_dataset(self, data: pd.DataFrame, meta: xb.Dataset) -> any:
        result = {
            'labels': (meta.target == meta.target_names[-1]).astype(int).to_numpy().reshape((-1,)),
            'class_names': meta.target_names,
            'ordinal_features': [i for i, label in (enumerate(meta.feature_names)) if label not
                                 in meta.categorical_features.keys()],
            'categorical_features': [i for i, label in (enumerate(meta.feature_names)) if label
                                     in meta.categorical_features.keys()],
            'categorical_names': {idx: [str(x) for x in meta.categorical_features[feature]] for idx, feature
                                  in enumerate(meta.feature_names) if feature in meta.categorical_features},
            'feature_names': meta.feature_names,
            'data': data.to_numpy()
        }
        
        for feature_idx in result['categorical_features']:
            feature_map = {feature: idx for idx, feature in enumerate(result['categorical_names'][feature_idx])}
            result['data'][:, feature_idx] = np.vectorize(lambda x: feature_map[str(x)])(result['data'][:, feature_idx])

        return result

    def inverse_transform_dataset(self, data: any, meta: xb.Dataset) -> pd.DataFrame:
        df = pd.DataFrame(data['data'], columns=meta.feature_names)
        for feature_idx in [i for i, label in (enumerate(meta.data.keys())) if label in meta.categorical_features.keys()]:
            df[meta.feature_names[feature_idx]] = df[meta.feature_names[feature_idx]].map(
                lambda entry: meta.categorical_features[meta.feature_names[feature_idx]][entry])
        return df

    def explain_instance(self, instance):
        """
        Creates an Anchor explanation based on a given instance

        :param instance: "Anchor" for explanation
        :param instance_set: textual information about subset for metric information
        :param threshold: Worst possible precision for the explanation
        :return: the explanation
        """

        instance = self.transform_dataset(instance, self.meta)

        self.explanation = self.explainer.explain_instance(instance['data'][0], self.predictor, threshold=self.min_precision)
        self.instance = instance['data'][0]
        return self.explanation

    @xb.prop
    def shape(self):
        return 'Hyperrectangle'

    @xb.prop
    def name(self):
        return 'Anchors'

    @xb.prop
    def instancing(self):
        return 'binary'

    @xb.metric
    def coverage(self):
        """
        The relative amount of data elements that are in the area of the explanation

        :return: the coverage value
        """
        if hasattr(self, 'explanation'):
            return self.explanation.coverage()

    @xb.metric
    def coverage_absolute(self):
        """
        Number of instances within the neighbourhood.
        """
        if hasattr(self, 'explanation'):
            try:
                return self.get_fit_anchor(self.anchors_dataset['data']).shape[0]
            except AttributeError:
                return 0

    @xb.metric
    def accuracy(self):
        """
        Relative amount of data elements in explanation neighborhood or given dataset that have the same explanation
        label as the label assigned by the ML model

        :return: the accuracy value
        """
        if hasattr(self, 'explanation'):
            explanation_label = self.explanation.exp_map["prediction"]
            neighborhood = self.anchors_dataset['data'][self.get_fit_anchor(self.anchors_dataset['data'])]
            if len(neighborhood) > 0:
                ml_pred = self.predictor(neighborhood)
                return np.count_nonzero(ml_pred == explanation_label) / len(neighborhood)
            else:
                return np.nan

    @xb.metric
    def accuracy_global(self):
        """
        The ML-accuracy of the explanation when applied to the whole dataset (not just the area of the explanation)

        :return: the precision value
        """
        if hasattr(self, 'explanation'):
            return self.explanation.precision()

    @xb.metric
    def balance_explanation(self):
        """
        Relative amount of data elements in the explanation neighborhood that had an assigned label value of 1
        (by the explanation)

        :return: the balance value
        """
        # balance is always 0 or 1 because Anchors creates a neighborhood where all elements are supposed to have
        # the same label as the one that was used to instantiate the explanation
        if hasattr(self, 'explanation'):
            return int(self.explanation.exp_map["prediction"])

    @xb.metric
    def balance_model(self):
        """
        Relative amount of data elements in the neighborhood of the explanation that had a label value of 1 assigned
        by the classification model

        :return: the balance value
        """
        if hasattr(self, 'explanation'):
            pred = self.predictor(self.anchors_dataset['data'])
            neighborhood = pred[self.get_fit_anchor(self.anchors_dataset['data'])]
            if neighborhood.size > 0:
                return np.mean(neighborhood)
            else:
                return np.nan

    @xb.metric
    def balance_data(self):
        """
        Relative amount of data elements in the neighborhood of the explanation with a label value of 1
        
        :return: the balance value
        """
        if hasattr(self, 'explanation'):
            neighborhood = self.anchors_dataset['labels'][self.get_fit_anchor(self.anchors_dataset['data'])]
            if neighborhood.size > 0:
                return np.mean(neighborhood)
            else:
                return np.nan

    @xb.metric
    def area_relative(self):
        """
        Relative amount of feature space over all features n that is specified by the explanation.
        area = Product[i=1->n] fi, f: 1 if feature is not in explanation, else 1/m, m: deminsionality of feature
        
        :return: the area value
        """
        if hasattr(self, 'explanation'):
            array = np.amax(self.anchors_dataset['data'], axis=0)[self.explanation.features()]
            array = array + 1
            return np.prod(1 / array)

    @xb.utility
    def get_fit_anchor(self, dataset):
        """
        Returns indices of data elements that are in the explanation neighborhood

        :param dataset: provided dataset
        :return: indices as numpy array
        """
        indices_categorical = np.where(np.all(dataset[:, self.explanation.features()] ==
                                              self.instance[self.explanation.features()], axis=1))[0]

        if np.size(indices_categorical) > 0:
            return indices_categorical

        # derive neighborhood from the name of the explanation
        try:
            index_lists = []
            for feature, name in zip(self.explanation.features(), self.explanation.names()):
                if ">=" in name:
                    index_lists.append(np.where(dataset[:, feature] >= float(name[name.index('>= ') + 3:])))
                elif "<=" in name:
                    index_lists.append(np.where(dataset[:, feature] <= float(name[name.index('<= ') + 3:])))
                elif ">" in name:
                    index_lists.append(np.where(dataset[:, feature] > float(name[name.index('> ') + 2:])))
                elif "<" in name:
                    index_lists.append(np.where(dataset[:, feature] < float(name[name.index('< ') + 2:])))
                elif "=" in name:
                    index_lists.append(np.where(dataset[:, feature] == name[name.index('" ') + 2:]))

            # intersect different sublits of indices
            indices_numerical = index_lists[0]
            for i in range(1, len(index_lists)):
                indices_numerical = np.intersect1d(indices_numerical, index_lists[i])

            if np.size(indices_numerical) < np.size(indices_categorical):
                return indices_categorical
            else:
                return indices_numerical
        except:
            return indices_categorical

    @xb.utility
    def get_explained_instance(self):
        """
        Returns instance that was explained

        :return: instance
        """
        return self.instance
