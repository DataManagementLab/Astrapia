from sklearn import metrics
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


def normalize(lst):
    # TODO there needs to be more work done here, metrics are currently normalized using all metrics from one explainer,
    #  instead of the same metrics from different explainers
    """
    Returns a sorted list of tuple (metric, value). Used to normalize non-relative metrics that is out of range [0,1].
    """
    new_lst = []
    max_val = max([i for _, i in lst])
    min_val = min([i for _, i in lst])
    for metric, score in lst:
        if not 0 <= score <= 1:
            temp = (score - min_val) / (max_val - min_val)
            new_lst.append((metric, round(temp, 4)))
        else:
            new_lst.append((metric, round(score, 4)))
    return sorted(new_lst, key=lambda x: x[0])


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

def visualize_table(name, pair):
    fig = go.Figure(data=[go.Table(
        header=dict(values=['metric', 'value'],
                    line_color='#bfbfbf',
                    fill_color='#e0e5df',
                    align='left'),
        cells=dict(values=[[metric for metric, _ in pair],  # 1st column
                           [round(value, 4) for _, value in pair]],  # 2nd column
                   line_color='#bfbfbf',
                   fill_color='#e0e5df',
                   align='left'))
    ])
    fig.update_layout(title_text=f'Metrics from explainer {name}')
    fig.show()
