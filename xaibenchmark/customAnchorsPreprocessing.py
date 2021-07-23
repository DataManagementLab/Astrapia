"""bla"""
# from __future__ import print_function
import copy
import sklearn
import numpy as np
import lime
import lime.lime_tabular
# import string
import os
import sys

class Bunch(object):
    """bla"""
    def __init__(self, adict):
        self.__dict__.update(adict)


def map_array_values(array, value_map):
    # value map must be { src : target }
    ret = array.copy()
    for src, target in value_map.items():
        ret[ret == src] = target
    return ret
def replace_binary_values(array, values):
    return map_array_values(array, {'0': values[0], '1': values[1]})

def load_dataset(dataframe, balance=False, discretize=True):


    feature_names = ["Age", "Workclass", "fnlwgt", "Education",
                     "Education-Num", "Marital Status", "Occupation",
                     "Relationship", "Race", "Sex", "Capital Gain",
                     "Capital Loss", "Hours per week", "Country", 'Income']
    features_to_use = [0, 1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    categorical_features = [1, 3, 5, 6, 7, 8, 9, 10, 11, 13]
    education_map = {
        '10th': 'Dropout', '11th': 'Dropout', '12th': 'Dropout', '1st-4th':
        'Dropout', '5th-6th': 'Dropout', '7th-8th': 'Dropout', '9th':
        'Dropout', 'Preschool': 'Dropout', 'HS-grad': 'High School grad',
        'Some-college': 'High School grad', 'Masters': 'Masters',
        'Prof-school': 'Prof-School', 'Assoc-acdm': 'Associates',
        'Assoc-voc': 'Associates',
    }
    occupation_map = {
        "Adm-clerical": "Admin", "Armed-Forces": "Military",
        "Craft-repair": "Blue-Collar", "Exec-managerial": "White-Collar",
        "Farming-fishing": "Blue-Collar", "Handlers-cleaners":
        "Blue-Collar", "Machine-op-inspct": "Blue-Collar", "Other-service":
        "Service", "Priv-house-serv": "Service", "Prof-specialty":
        "Professional", "Protective-serv": "Other", "Sales":
        "Sales", "Tech-support": "Other", "Transport-moving":
        "Blue-Collar",
    }
    country_map = {
        'Cambodia': 'SE-Asia', 'Canada': 'British-Commonwealth', 'China':
        'China', 'Columbia': 'South-America', 'Cuba': 'Other',
        'Dominican-Republic': 'Latin-America', 'Ecuador': 'South-America',
        'El-Salvador': 'South-America', 'England': 'British-Commonwealth',
        'France': 'Euro_1', 'Germany': 'Euro_1', 'Greece': 'Euro_2',
        'Guatemala': 'Latin-America', 'Haiti': 'Latin-America',
        'Holand-Netherlands': 'Euro_1', 'Honduras': 'Latin-America',
        'Hong': 'China', 'Hungary': 'Euro_2', 'India':
        'British-Commonwealth', 'Iran': 'Other', 'Ireland':
        'British-Commonwealth', 'Italy': 'Euro_1', 'Jamaica':
        'Latin-America', 'Japan': 'Other', 'Laos': 'SE-Asia', 'Mexico':
        'Latin-America', 'Nicaragua': 'Latin-America',
        'Outlying-US(Guam-USVI-etc)': 'Latin-America', 'Peru':
        'South-America', 'Philippines': 'SE-Asia', 'Poland': 'Euro_2',
        'Portugal': 'Euro_2', 'Puerto-Rico': 'Latin-America', 'Scotland':
        'British-Commonwealth', 'South': 'Euro_2', 'Taiwan': 'China',
        'Thailand': 'SE-Asia', 'Trinadad&Tobago': 'Latin-America',
        'United-States': 'United-States', 'Vietnam': 'SE-Asia'
    }
    married_map = {
        'Never-married': 'Never-Married', 'Married-AF-spouse': 'Married',
        'Married-civ-spouse': 'Married', 'Married-spouse-absent':
        'Separated', 'Separated': 'Separated', 'Divorced':
        'Separated', 'Widowed': 'Widowed'
    }
    label_map = {'<=50K': 'Less than $50,000', '>50K': 'More than $50,000'}

    def cap_gains_fn(x):
        x = x.astype(float)
        d = np.digitize(x, [0, np.median(x[x > 0]), float('inf')],
                        right=True).astype('|S128')
        return map_array_values(d, {'0': 'None', '1': 'Low', '2': 'High'})

    transformations = {
        3: lambda x: map_array_values(x, education_map),
        5: lambda x: map_array_values(x, married_map),
        6: lambda x: map_array_values(x, occupation_map),
        10: cap_gains_fn,
        11: cap_gains_fn,
        13: lambda x: map_array_values(x, country_map),
        # 14: lambda x: map_array_values(x, label_map),
    }
    dataset = load_csv_dataset(
        dataframe, -1, ', ',
        feature_names=feature_names, features_to_use=features_to_use,
        categorical_features=categorical_features, discretize=discretize,
        balance=balance, feature_transformations=transformations)

    return dataset


def load_csv_dataset(data, target_idx, delimiter=',',
                     feature_names=None, categorical_features=None,
                     features_to_use=None, feature_transformations=None,
                     discretize=False, balance=False, fill_na='-1', filter_fn=None, skip_first=False):
    """if not feature names, takes 1st line as feature names
    if not features_to_use, use all except for target
    if not categorical_features, consider everything < 20 as categorical"""
    if feature_transformations is None:
        feature_transformations = {}

    if target_idx < 0:
        target_idx = data.shape[1] + target_idx
    ret = Bunch({})
    if feature_names is None:
        feature_names = list(data[0])
        data = data[1:]
    else:
        feature_names = copy.deepcopy(feature_names)
    if skip_first:
        data = data[1:]
    if filter_fn is not None:
        data = filter_fn(data)
    for feature, fun in feature_transformations.items():
        data[:, feature] = fun(data[:, feature])
    labels = data[:, target_idx]
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(labels)
    ret.labels = le.transform(labels)
    labels = ret.labels
    ret.class_names = list(le.classes_)
    ret.class_target = feature_names[target_idx]
    if features_to_use is not None:
        data = data[:, features_to_use]
        feature_names = ([x for i, x in enumerate(feature_names)
                          if i in features_to_use])
        if categorical_features is not None:
            categorical_features = ([features_to_use.index(x)
                                     for x in categorical_features])
    else:
        data = np.delete(data, target_idx, 1)
        feature_names.pop(target_idx)
        if categorical_features:
            categorical_features = ([x if x < target_idx else x - 1
                                     for x in categorical_features])

    if categorical_features is None:
        categorical_features = []
        for f in range(data.shape[1]):
            if len(np.unique(data[:, f])) < 20:
                categorical_features.append(f)
    categorical_names = {}
    for feature in categorical_features:
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(data[:, feature])
        data[:, feature] = le.transform(data[:, feature])
        categorical_names[feature] = le.classes_
    data = data.astype(float)
    ordinal_features = []
    if discretize:
        disc = lime.lime_tabular.QuartileDiscretizer(data,
                                                     categorical_features,
                                                     feature_names)
        data = disc.discretize(data)
        ordinal_features = [x for x in range(data.shape[1])
                            if x not in categorical_features]
        categorical_features = list(range(data.shape[1]))
        categorical_names.update(disc.names)
    for x in categorical_names:
        categorical_names[x] = [y.decode() if type(y) == np.bytes_ else y for y in categorical_names[x]]
    ret.ordinal_features = ordinal_features
    ret.categorical_features = categorical_features
    ret.categorical_names = categorical_names
    ret.feature_names = feature_names
    np.random.seed(1)
    if balance:
        idxs = np.array([], dtype='int')
        min_labels = np.min(np.bincount(labels))
        for label in np.unique(labels):
            idx = np.random.choice(np.where(labels == label)[0], min_labels)
            idxs = np.hstack((idxs, idx))
        data = data[idxs]
        labels = labels[idxs]
        ret.data = data
        ret.labels = labels

    ret.data = data
    return ret.data[ret.data.shape[0]-1]
