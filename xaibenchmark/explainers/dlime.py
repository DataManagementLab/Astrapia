import numpy as np
from xaibenchmark import preprocessing
from xaibenchmark.explainer import Explainer
from xaibenchmark.dlime.explainer_tabular import LimeTabularExplainer
import xaibenchmark as xb
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LinearRegression


class DLimeExplainer(Explainer):

    def __init__(self, data, predict_fn, discretize_continuous=True):
        self.train = data.train
        self.test = data.test

        self.explainer = LimeTabularExplainer(self.train,
                                             mode="classification",
                                             feature_names=data.feature_names,
                                             class_names=data.target_names,
                                             discretize_continuous=True,
                                             verbose=False)

        clustering = AgglomerativeClustering().fit(self.train)
        self.clustered_data = np.column_stack([self.train, clustering.labels_])

        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.train)
        distances, self.indices = nbrs.kneighbors(self.test)
        self.clabel = clustering.labels_

        self.predict_fn = predict_fn
        self.kernel_width = np.sqrt(data.train.shape[1]) * .75

    def explain_instance(self, x, num_features=10):

        p_label = self.clabel[self.indices[x]]
        N = self.clustered_data[self.clustered_data[:, 30] == p_label]
        subset = np.delete(N, 30, axis=1)
        self.instance = self.test[x]

        self.explanation = self.explainer.explain_instance_hclust(self.instance,
                                                                             self.predict_fn.predict_proba,
                                                                             num_features=num_features,
                                                                             model_regressor=LinearRegression(),
                                                                             clustered_data=subset,
                                                                             regressor='linear',
                                                                             labels=(0, 1))
        return self.explanation

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
        pass

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
    def balance_explanation(self):
        pass

    @xb.utility
    def distance(self, x, y):
        return np.linalg.norm(x - y)

    @xb.utility
    def get_weighted_instances(self):
        pass

    @xb.utility
    def get_explained_instance(self):
        return self.instance

    @xb.utility
    def get_training_data(self):
        return self.train

    @xb.utility
    def predict_instance_surrogate(self, instance):
        pass