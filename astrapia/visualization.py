import json
import textwrap
import plotly.graph_objects as go


def normalize(dicts, relevant_metrics):
    """
    Normalize non-relative metric values in order to visualize them side-by-side with relative metrics.
    If there is only one value for a non-relative metric, it is left out of the result.

    :param dicts: dictionary of metrics
    :param relevant_metrics: list of metric names that should be normalized if they are non-relative
    :return: dictionary of metrics with adjusted values
    """

    # initialize dictionary that will be returned
    res = {name: {} for name, metrics in dicts.items()}

    # detect metric names that consist non-relative values
    critical_metrics = []
    for name, metrics in dicts.items():
        for metric in relevant_metrics:
            if metric in metrics:
                if not 0 <= metrics[metric] <= 1:
                    critical_metrics.append(metric)
    critical_metrics = list(set(critical_metrics))

    for metric in relevant_metrics:
        if metric in critical_metrics:

            # accumulate all values of a non-relative metric
            current_values = []
            for name, metrics in dicts.items():
                if metric in metrics:
                    current_values.append((name, metrics[metric]))

            # if there is more than one value, normalize the values to [0, 1] based on the highest and lowest values
            if len(current_values) > 1:
                max_val = max([i for _, i in current_values])
                min_val = min([i for _, i in current_values])
                if max_val == min_val:
                    for name, _ in current_values:
                        res[name][metric + '*'] = 1
                else:
                    for name, value in current_values:
                        res[name][metric+'*'] = (value - min_val) / (max_val - min_val)

        else:
            for name in dicts.keys():
                if metric in dicts[name]:
                    m, v = normalize_balance(metric, dicts[name][metric])
                    res[name][m] = v

    return res


def fill_in_value(metric_dict, metric, numeric=True):
    """
    Given the name of a metric and a dict that may have a value for it, return the rounded value or a dash sign in case
    the metric does not exist in the dict

    :param numeric: information whether numeric values are handled
    :param metric_dict: dictionary of metrics and their respective value
    :param metric: name of the metric
    :return: metric value as String
    """
    if metric in metric_dict:
        if not numeric:
            return metric_dict[metric]
        else:
            return round_table_value(metric_dict[metric])
    else:
        return '-'


def round_table_value(value):
    """
    Given a value, return it as rounded string

    :param value: metric value
    :return: rounded value as String
    """
    str_value = str(value)
    if 'e' in str_value:
        return str_value[:6] + str_value[str_value.index('e'):]
    else:
        return round(value, 6)


def normalize_balance(metric, value):
    """
    Balance-related values are normalised to [0,1], having a higher value the closer they are to 0.5

    :param metric: name of the metric
    :param value: value of the metric
    :return: metric name, normalized metric value
    """
    if 'balance' in metric:
        return metric + "*", 1 - 2 * abs(value - 0.5)
    else:
        return metric, value


def load_metrics_from_json(path):
    """
    Load metric data from a file where metric data has been stored

    :param path: path to .json file
    :return: metric data as dictionary
    """
    with open(path) as json_file:
        return json.load(json_file)


def print_properties(data):
    """
    prints properties of explainers as table

    :param data: data as returned by Comparator
    """

    headline = ['Property']
    values = []
    properties = [x.keys() for x in data['properties'].values()]
    all_properties = sorted(list(set([item for sublist in properties for item in sublist])))

    for name, properties in data['properties'].items():
        headline += [name]
        explainer_values = [fill_in_value(properties, prop, False) for prop in all_properties]
        values.append(explainer_values)

    fig = go.Figure(data=[go.Table(
        header=dict(values=headline,
                    line_color='#bfbfbf',
                    fill_color='#e0e5df',
                    align='left'),
        cells=dict(values=[all_properties] + values,  # 2nd column
                   line_color='#bfbfbf',
                   fill_color='#e0e5df',
                   align='left'))
    ])
    fig.update_layout(title_text=f'Properties from all explainers')
    fig.show()


def print_metrics(data, explainer=None, index=None, plot='table', show_metric_with_one_value=True):
    """
    Output metrics of the explanations on the console. Either averaged metrics or metrics from single explanations

    :param data: dictionary with metrics data
    :param explainer: Optional, in case you only want metrics from one explainer
    :param index: Optional, in case you only want metrics from one explanation
    :param plot: Optional. Visualize the metrics in bar chart or table form. Options: 'bar' or 'table'
    :param show_metric_with_one_value: show metric even if just one of the explainers has a value for it
    """

    assert plot == 'bar' or plot == 'table', 'Wrong input. Check again.'

    # Defining names and fetching data depending on the input parameters
    if explainer is not None:
        if index is not None:
            output = data['separate_metrics'][explainer][str(index)]
            header = "Metrics of the explanation with index " + str(index) + " from explainer " + explainer
        else:
            output = data['averaged_metrics'][explainer]
            header = "Average Metrics from explainer " + explainer

        pair = sorted(list(zip(output.keys(), output.values())), key=lambda tup: tup[0])

        # Visualize the table
        if plot == 'table':
            fig = go.Figure(data=[go.Table(
                header=dict(values=['metric', 'value'],
                            line_color='#bfbfbf',
                            fill_color='#e0e5df',
                            align='left'),
                cells=dict(values=[[metric for metric, _ in pair],  # 1st column
                                   [round_table_value(value) for _, value in pair]],  # 2nd column
                           line_color='#bfbfbf',
                           fill_color='#e0e5df',
                           align='left'))
            ])
            fig.update_layout(title_text=header)
            fig.show()

        # Visualize the plot and normalize balance
        elif plot == 'bar':
            normalized_pair = [normalize_balance(metric, value) for (metric, value) in pair if 0 <= value <= 1]
            fig = go.Figure([go.Bar(x=[metric for metric, _ in normalized_pair],
                                    y=[value for _, value in normalized_pair],
                                    text=[round(value, 3) for _, value in normalized_pair],
                                    textposition='outside')])
            fig.update_layout(title_text=header, plot_bgcolor='#fffaf4', height=600, margin=dict(l=20, r=20, t=60, b=200))
            annotation = textwrap.wrap("Non-relative metrics are not shown here because they cannot be compared with "
                                       "relative values in the same plot. Balance-related metrics were rescaled to "
                                       "display how close they are to 0.5, because that is the optimal balance value.")
            annotation = "<br>".join(annotation)
            fig.add_annotation(dict(font=dict(color='black', size=15), x=0, y=-0.51, showarrow=False,
                                    text=annotation,
                                    textangle=0, xanchor='left', xref="paper", yref="paper"))
            fig.update_annotations(align="left")
            fig.show()

    else:
        # Retrieve all metrics from all explainers
        all_explainers = [x.keys() for x in data['averaged_metrics'].values()]
        relevant_metrics = sorted(list(set([item for sublist in all_explainers for item in sublist])))

        # Filter metrics that only exist for one explainer
        if not show_metric_with_one_value:
            filtered_metrics = []
            for metric in relevant_metrics:
                count = 0
                for name, metrics in data['averaged_metrics'].items():
                    if metric in metrics:
                        count += 1
                if count > 1:
                    filtered_metrics.append(metric)
            relevant_metrics = filtered_metrics

        # Visualize table
        if plot == 'table':
            headline = ['Metric']
            values = []

            for name, metrics in data['averaged_metrics'].items():
                headline += [name]
                explainer_values = [fill_in_value(metrics, metric) for metric in relevant_metrics]
                values.append(explainer_values)

            fig = go.Figure(data=[go.Table(
                header=dict(values=headline,
                            line_color='#bfbfbf',
                            fill_color='#e0e5df',
                            align='left'),
                cells=dict(values=[relevant_metrics] + values,  # 2nd column
                           line_color='#bfbfbf',
                           fill_color='#e0e5df',
                           align='left'))
            ])
            fig.update_layout(title_text=f'Metrics from all explainers')
            fig.show()

        # Visualize bar chart
        elif plot == 'bar':
            # Defining a custom colorblind-friendly color palette
            # Taken from https://jacksonlab.agronomy.wisc.edu/2016/05/23/15-level-colorblind-friendly-palette/
            colors = ["#004949", "#ffff6d", "#009292", "#db6d00", "#490092", "#ff6db6", "#ffb6db", "#920000",
                      "#006ddb", "#b6dbff", "#b66dff", "#24ff24", "#6db6ff"]
            col = {n: c for (n, c) in zip(data["explainers"], colors[:len(data["explainers"])])}

            # Normalizing metrics
            normalized_metrics = [(name, sorted(list(zip(metrics.keys(), metrics.values())),
                                                key=lambda tup: tup[0]))
                                  for name, metrics in normalize(data['averaged_metrics'],
                                                                 relevant_metrics).items()]

            fig = go.Figure(data=[go.Bar(name=name,
                                         x=[metric for metric, _ in normalized_pair],
                                         y=[value for _, value in normalized_pair],
                                         text=[round(value, 3) for _, value in normalized_pair],
                                         textposition='outside',
                                         marker={'color': col[name]})
                                  for (name, normalized_pair) in normalized_metrics])
            fig.update_layout(barmode='group', title_text=f'Metric Comparison', plot_bgcolor='#ececea',
                              height=600, margin=dict(l=20, r=20, t=60, b=200))

            annotation = textwrap.wrap("Non-relative metrics are normalized to be between 0 and 1. If there is only "
                                       "one value for a non-relative metric (in case metrics with only a single value "
                                       "are shown), then this metric is not visible. Balance-related metrics were "
                                       "rescaled to display how close they are to 0.5, because that is the optimal "
                                       "balance value.")
            annotation = "<br>".join(annotation)
            fig.add_annotation(dict(font=dict(color='black', size=15), x=0, y=-0.52, showarrow=False,
                                    text=annotation,
                                    textangle=0, xanchor='left', xref="paper", yref="paper"))
            fig.update_annotations(align="left")

            fig.show()
