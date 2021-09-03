from sklearn import metrics
import pandas as pd
import xaibenchmark as xb


def model_properties(y_test, modelpredictions, labels=[]):
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
