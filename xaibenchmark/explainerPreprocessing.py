import copy
import sklearn
import numpy as np
import lime
import lime.lime_tabular
import pandas as pd


def lime_preprocess_datasets(data_df, categorical_features, keys):
    """
    Preprocesses multiple datasets to be used by LIME and a ML model that can be used by LIME
    :param data_df: list of datasets
    :param categorical_features: categorical features of the dataset
    :param keys: keys of the dataset
    :return: list of preprocessed datasets
    """
    return [lime_preprocess_dataset(df, categorical_features, keys) for df in data_df]


def lime_preprocess_dataset(df, categorical_features, keys):
    """
    Preprocesses a dataset to be used by LIME
    :param df: dataset
    :param categorical_features: categorical features of the dataset
    :param keys: keys of the dataset
    :return: preprocessed dataset
    """
    cat_df = pd.get_dummies(df, columns=categorical_features.keys())
    missing_cols = {cat + '_' + str(attr) for cat in categorical_features
                    for attr in categorical_features[cat]} - set(cat_df.columns)
    for c in missing_cols:
        cat_df[c] = 0

    cont_idx = list(set(keys) - set(categorical_features.keys()))
    cat_idx = [cat + '_' + str(attr) for cat in categorical_features
               for attr in categorical_features[cat]]
    idx = cont_idx + cat_idx
    return cat_df[idx]


def anchors_preprocess_instance(data):
    """
    customized data preprocessing from the Anchors library that takes a dataset with an additional instance at the end
    of it and preprocesses the set in order to get a preprocessed representation of the additional instance
    :param data: adult dataset with one interesting instance at the end of it
    :return: preprocessed version of the original additional instance
    """

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

    def cap_gains_fn(x):
        x = x.astype(float)
        d = np.digitize(x, [0, np.median(x[x > 0]), float('inf')],
                        right=True).astype('|S128')
        return map_array_values(d, {'0': 'None', '1': 'Low', '2': 'High'})

    feature_transformations = {
        3: lambda x: map_array_values(x, education_map),
        5: lambda x: map_array_values(x, married_map),
        6: lambda x: map_array_values(x, occupation_map),
        10: cap_gains_fn,
        11: cap_gains_fn,
        13: lambda x: map_array_values(x, country_map),
    }

    target_idx = data.shape[1] - 1
    ret = Bunch({})
    feature_names = copy.deepcopy(feature_names)

    for feature, fun in feature_transformations.items():
        data[:, feature] = fun(data[:, feature])
    labels = data[:, target_idx]
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(labels)
    ret.labels = le.transform(labels)
    ret.class_names = list(le.classes_)
    ret.class_target = feature_names[target_idx]
    if features_to_use is not None:
        data = data[:, features_to_use]
        feature_names = ([x for i, x in enumerate(feature_names)
                          if i in features_to_use])
        if categorical_features is not None:
            categorical_features = ([features_to_use.index(x)
                                     for x in categorical_features])

    categorical_names = {}
    for feature in categorical_features:
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(data[:, feature])
        data[:, feature] = le.transform(data[:, feature])
        categorical_names[feature] = le.classes_
    data = data.astype(float)


    # Discretization
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

    ret.data = data
    return ret.data[ret.data.shape[0]-1]


# Content taken from Anchors library


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
