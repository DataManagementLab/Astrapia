from sklearn import metrics
import pandas as pd
import xaibenchmark as xb
import plotly.graph_objects as go
import copy

def model_properties( y_test, modelpredictions, labels=[]):
    """
    Returns a dictionary containing accuracy, precision, f1-score, etc.
    """
    
    if len(y_test) != len(modelpredictions) :
        print("The length of the labels array needs to be the same length as the prediction array")
        return
    if not labels: 
        return metrics.classification_report(y_test, modelpredictions, output_dict=True)
    else:
        return metrics.classification_report(y_test, modelpredictions, labels=labels, output_dict=True)

def onehot_encode(data: pd.DataFrame, meta: xb.Dataset) -> any:
    """
    One-hot encodes the dataframe.

    :param data: DataFrame to be encoded
    :param meta: Astrapia Dataset metadata with categorical_features attribute
    :return: One-hot encoded DataFrame
    """

    transformed_df = pd.DataFrame(index=data.index)

    for feature in set(data.columns) - set(meta.categorical_features):
        transformed_df[feature] = data[feature]

    for feature in meta.categorical_features:
        for label in meta.categorical_features[feature]:
            transformed_df[feature+'_'+str(label)] = (data[feature]==label).astype(int)

    return transformed_df

# def normalize(lst):
#     """
#     Returns a sorted list of tuple (metric, value). Used to normalize non-relative metrics that is out of range [0,1].
#     """
#     new_lst = []
#     max_val = max([i for _, i in lst])
#     min_val = min([i for _, i in lst])
#     for metric, score in lst:
#         if not 0 <= score <= 1:
#             temp = (score - min_val) / (max_val - min_val)
#             new_lst.append((metric, round(temp, 4)))
#         else:
#             new_lst.append((metric, round(score, 4)))
#     return sorted(new_lst, key=lambda x: x[0])


def normalize2(dicts):
    res = copy.deepcopy(dicts)
    criticalmetrics = []
    for name, metrics in dicts.items():
        for k, v in metrics.items():
            if not 0 <= v <= 1:
                criticalmetrics.append(k)
    criticalmetrics = list(set(criticalmetrics))

    for crit in criticalmetrics:
        currentvalues = []
        for name, metrics in dicts.items():
            if crit in metrics:
                currentvalues.append((name, metrics[crit]))

        if len(currentvalues) == 1:
            res[currentvalues[0][0]][crit] = 1
        else:
            max_val = max([i for _, i in currentvalues])
            min_val = min([i for _, i in currentvalues])
            for name, value in currentvalues:
                res[name][crit] = (value - min_val) / (max_val - min_val)

    return res


def fill_in_value(metric_dict, metric):
    if metric in metric_dict:
        return round(metric_dict[metric], 6)
    else:
        return '-'
