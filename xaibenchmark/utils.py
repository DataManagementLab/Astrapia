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
