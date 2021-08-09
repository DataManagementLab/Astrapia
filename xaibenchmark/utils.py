from sklearn import metrics

def model_properties( y_test, modelpredictions, labels=[]):
    """
    Returns a dictionary containing accuracy, precision, f1-score, etc.
    """
    
    if(len(y_test) != len(modelpredictions)) :
        print("The length of the labels array needs to be the same length as the prediction array")
        return
    if not labels: 
        return metrics.classification_report(y_test, modelpredictions, output_dict=True)
    else:
        return metrics.classification_report(y_test, modelpredictions, labels=labels, output_dict=True)


def normalize(lst):
    """
    only normalize non-relative metrics, which is out of the [0,1] range
    """
    new_lst = []
    max_val = max([i for _, i in lst])
    min_val = min([i for _, i in lst])
    for metric, score in lst:
        if not 0<score<1:
            temp = (score - min_val) / (max_val - min_val)
            new_lst.append((metric, round(temp, 4)))
        else:
            new_lst.append((metric, round(score, 4)))
    return sorted(new_lst, key=lambda x: x[0])
