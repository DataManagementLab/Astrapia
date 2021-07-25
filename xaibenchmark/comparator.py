from xaibenchmark import explainer, preprocessing
import sklearn.ensemble
from anchor import utils
from xaibenchmark import load_adult as la
import xaibenchmark.explainers as explainers


class ExplainerComparator:

    def __init__(self):
        self.explainers = {}
        self.current_metrics = {}
        self.current_explanations = {}

    def add_explainer(self, explainer, name):
        self.explainers[name] = explainer

    def explain_instance(self, instance):
        for name, explainer in self.explainers.items():
            self.current_explanations[name] = explainer.explain_instance(instance)
            explainer.report()
            explainer.infer_metrics(printing=False)
        self.current_metrics = {name: explainer.report() for name, explainer in self.explainers.items()}

    def print_metrics(self):
        for name, metrics in self.current_metrics.items():
            print(name, ":", metrics)


if __name__ == "__main__":
    dataset_folder = '../data/'
    data = la.load_csv_data('adult', root_path=dataset_folder)

    anchor_training_set = utils.load_dataset('adult', balance=True, dataset_folder=dataset_folder, discretize=True)
    anchor_ml_model = sklearn.ensemble.RandomForestClassifier(n_estimators=50, n_jobs=5)
    anchor_ml_model.fit(anchor_training_set.train, anchor_training_set.labels_train)
    anchor1 = explainers.AnchorsExplainer(anchor_ml_model, dataset_folder, 'adult')

    lime_training_set = preprocessing.lime_preprocess_dataset(data.data, data.categorical_features, data.data.keys())
    lime_ml_model = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
    lime_ml_model.fit(lime_training_set, data.target.to_numpy().reshape(-1))
    lime1 = explainers.LimeExplainer(data, lime_ml_model, discretize_continuous=False)

    x = data.data.iloc[[5000]]

    ourFirstCollector = ExplainerComparator()
    ourFirstCollector.add_explainer(anchor1, "ANCHORS")
    ourFirstCollector.add_explainer(lime1, "LIME")
    ourFirstCollector.explain_instance(data.data.iloc[[5000]])
    ourFirstCollector.print_metrics()