from xaibenchmark import explainer
import sklearn.ensemble
from anchor import utils
from xaibenchmark import load_adult as la
import pandas as pd


class ExplainerComparator:

    def __init__(self):
        self.explainers = {}
        self.current_metrics = {}
        self.current_explanations = {}

    def add_explainer(self, explainer, name):
        self.explainers[name] = explainer

    def explain_instance(self, instance):
        self.current_explanations = {name: explainer.explain_instance(instance) for name, explainer in self.explainers.items()}
        for _, explainer in self.explainers.items():
            explainer.infer_metrics(printing=False)
        self.current_metrics = {name: explainer.report() for name, explainer in self.explainers.items()}

    def print_metrics(self):
        for name, metrics in self.current_metrics.items():
            print(name, ":", metrics)


data = la.load_csv_data('adult', root_path='../data')



# make sure you have adult/adult.data inside dataset_folder
dataset_folder = '../data/'
adult_dataset = utils.load_dataset('adult', balance=True, dataset_folder=dataset_folder, discretize=True)

rf = sklearn.ensemble.RandomForestClassifier(n_estimators=50, n_jobs=5)
rf.fit(adult_dataset.train, adult_dataset.labels_train)
anchor1 = explainer.AnchorsExplainer(rf, dataset_folder)


def preprocess(*data_df):
    def process_single(df):
        cat_df = pd.get_dummies(df, columns=data.categorical_features.keys())
        missing_cols = {cat + '_' + str(attr) for cat in data.categorical_features \
                        for attr in data.categorical_features[cat]} - set(cat_df.columns)
        for c in missing_cols:
            cat_df[c] = 0

        cont_idx = list(set(data.data.keys()) - set(data.categorical_features.keys()))
        cat_idx = [cat + '_' + str(attr) for cat in data.categorical_features \
                   for attr in data.categorical_features[cat]]
        idx = cont_idx + cat_idx
        return cat_df[idx]

    # Preprocess function for one-hot encoding categorical data
    return [process_single(df) for df in data_df]


train, dev, test = preprocess(data.data, data.data_dev, data.data_test)
labels_train, labels_dev, labels_test = data.target, data.target_dev, data.target_test
rf2 = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
rf2.fit(train, labels_train.to_numpy().reshape(-1))
lime1 = explainer.LimeExplainer(data, rf2, discretize_continuous=False)

x = data.data.iloc[[5000]]

ourfirstcollector = ExplainerComparator()
ourfirstcollector.add_explainer(anchor1, "ANCHORS")
ourfirstcollector.add_explainer(lime1, "LIME")
ourfirstcollector.explain_instance(data.data.iloc[[5000]])
ourfirstcollector.print_metrics()