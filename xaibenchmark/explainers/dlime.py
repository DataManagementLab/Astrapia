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