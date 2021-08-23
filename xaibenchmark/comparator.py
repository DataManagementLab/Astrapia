from collections import defaultdict
from xaibenchmark.utils import normalize
from xaibenchmark.utils import normalize2
from xaibenchmark.utils import visualize_table
import plotly.graph_objects as go


class ExplainerComparator:

    def __init__(self):
        # Dictionary with key: name of explainer, value: explainer as object
        self.explainers = {}

        # Dictionary with key: name of explainer, value: dictionary with key: name of metric, value: average value
        self.averaged_metrics = {}

        # Dictionary with key: name of explainer, value: dictionary with key: index of explanation,
        # value: explanation as object
        self.explanations = {}

        # Dictionary with key: name of explainer, value: dictionary with key: index of explanation,
        # value: dictionary with key: name of metric, value: metric value
        self.metrics = {}

        # instances that are used to create explanations
        self.instances = None

    def add_explainer(self, explainer, name):
        """
        Add an instantiated explainer to the comparator
        :param explainer: explainer as python object
        :param name: name that is supposed to be used for the explainer
        """
        self.explainers[name] = explainer

    def explain_instances(self, instances):
        """
        Create explanations for all combinations of provided explainers and instances, then save metrics
        :param instances: instances to be used to create explanations as pandas dataframe
        """

        # Reset aggregation attributes
        self.averaged_metrics = {}
        self.explanations = {}
        self.metrics = {}
        self.instances = instances

        # Iterate over all explainers
        for name, explainer in self.explainers.items():
            explainer_average_metrics = defaultdict(float)
            aggregated_explainer_metrics = {}
            aggregated_explanations = {}

            # iterate over all instances
            for index in range(0, instances.shape[0]):
                explanation = explainer.explain_instance(instances.iloc[[index]])
                explainer.report()
                explainer.infer_metrics(printing=False)

                explanation_metrics = {}
                for (metric, value) in explainer.report():
                    # add up metrics in order to compute average values later
                    explainer_average_metrics[metric] += value
                    # save metrics of the explanation separately
                    explanation_metrics[metric] = value

                aggregated_explainer_metrics[index] = explanation_metrics
                aggregated_explanations[index] = explanation

            # compute average metrics by dividing added metrics by the amount of provided instances
            explainer_average_metrics = {metric: value/instances.shape[0]
                                         for metric, value in explainer_average_metrics.items()}

            self.averaged_metrics[name] = explainer_average_metrics
            self.metrics[name] = aggregated_explainer_metrics
            self.explanations[name] = aggregated_explanations

    def print_metrics(self, explainer=None, index=None, plot=None):
        """
        Output metrics of the explanations on the console. Either averaged metrics or metrics from single explanations
        :param explainer: Optional, in case you only want metrics from one explainer
        :param index: Optional, in case you only want metrics from one explanation
        :param plot: Optional. Visualize the metrics in bar chart or table form. Options: 'bar' or 'table'
        """

        assert plot is None or plot == 'bar' or plot == 'table', 'Wrong input. Check again.'

        if explainer is not None:
            if index is not None:
                output = self.metrics[explainer][index]
                print("Metric values for explainer", explainer,
                      ", explanation created with the", index, "-th instance of the given data")
            else:
                output = self.averaged_metrics[explainer]
                print("Average metric values for the explainer", explainer, ":")

            for metric, value in output.items():
                print("\t", metric, ":", value)

            pair = list(zip(output.keys(), output.values()))

            if plot == 'bar':
                normalized_pair = normalize(pair)
                fig = go.Figure([go.Bar(x=[metric for metric, _ in normalized_pair],
                                        y=[value for _, value in normalized_pair],
                                        marker_color='#96a48b')])

                title = '' if index else 'Average'
                fig.update_layout(title_text=f'{title} Metric Comparison using {explainer}',
                                  plot_bgcolor='#fffaf4')
                fig.show()
            elif plot == 'table':
                visualize_table(explainer, pair)

        else:
            for name, metrics in self.averaged_metrics.items():
                # print("Average metric values for explainer", name, ":")
                # for metric, value in metrics.items():
                #     print("\t", metric, ":", value)

                if plot == 'table':
                    pair = list(zip(metrics.keys(), metrics.values()))
                    visualize_table(name, pair)

            if plot == 'bar':
                normalized_metrics = [(name, normalize(list(zip(metrics.keys(), metrics.values()))))
                                      for name, metrics in self.averaged_metrics.items()]
                normalized2_metrics = [(name, (list(zip(metrics.keys(), metrics.values()))))
                                       for name, metrics in normalize2(self.averaged_metrics).items()]
                # original metrics
                # print([(name, list(zip(metrics.keys(), metrics.values())))
                #        for name, metrics in self.averaged_metrics.items()])
                # metrics normalized
                # print(normalized_metrics)
                # metrics normalized with new normalization
                # print(sorted(normalized2_metrics, key=lambda x: x[0]))

                # original normalization
                fig = go.Figure(data=[go.Bar(name=name,
                                             x=[metric for metric, _ in normalized_pair],
                                             y=[value for _, value in normalized_pair])
                                      for (name, normalized_pair) in normalized_metrics])
                fig.update_layout(barmode='group', title_text=f'Metric Comparison original normalization', plot_bgcolor='#ececea')
                fig.show()

                # new normalization
                fig = go.Figure(data=[go.Bar(name=name,
                                             x=[metric for metric, _ in normalized_pair],
                                             y=[value for _, value in normalized_pair])
                                      for (name, normalized_pair) in normalized2_metrics])
                fig.update_layout(barmode='group', title_text=f'Metric Comparison new normalization', plot_bgcolor='#ececea')
                fig.show()

