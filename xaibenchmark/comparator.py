from collections import defaultdict
from xaibenchmark.samplers import base_sampler, random, splime
from xaibenchmark.utils import normalize2
from xaibenchmark.utils import fill_in_value
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

    def explain_representative(self, data, sampler='random', count=10, verbose=False):
        """
        Create a representative explanation for the given data
        :param data: pandas dataframe with the data to be explained
        :param sampler: sampler to be used to create representative explanation
        :param count: amount of representative samples to be created
        :param verbose: print details of the created explanation       
        """

        # Map sampler names to objects
        # Add new samplers here
        sampler_map = {
            'random': random.RandomSampler,
            'splime': splime.SPLimeSampler,
        }
        if issubclass(type(sampler), base_sampler.Sampler):
            # sampler is an object
            sampler = sampler()
        elif sampler in sampler_map:
            # sampler is a string
            sampler = sampler_map[sampler]()
        else:
            # sampler is unknown
            raise Exception('Invalid sampler \''+sampler+'\'')

        # sampler is now a Sampler object
        # sample instances and explain them
        instances = sampler.sample(data, count)
        
        if verbose:
            print('---- sampled representative instanes ----')
            print(instances)

        return self.explain_instances(instances)
        

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
                header = "Metrics of the explanation with index " + str(index) + " from explainer " + explainer
            else:
                output = self.averaged_metrics[explainer]
                header = "Average Metrics from explainer " + explainer

            pair = sorted(list(zip(output.keys(), output.values())), key=lambda tup: tup[0])

            if plot == 'table':
                fig = go.Figure(data=[go.Table(
                    header=dict(values=['metric', 'value'],
                                line_color='#bfbfbf',
                                fill_color='#e0e5df',
                                align='left'),
                    cells=dict(values=[[metric for metric, _ in pair],  # 1st column
                                       [round(value, 6) for _, value in pair]],  # 2nd column
                               line_color='#bfbfbf',
                               fill_color='#e0e5df',
                               align='left'))
                ])
                fig.update_layout(title_text=header)
                fig.show()

            elif plot == 'bar':
                normalized_pair = [(metric, value) for (metric, value) in pair if 0 <= value <= 1]
                fig = go.Figure([go.Bar(x=[metric for metric, _ in normalized_pair],
                                        y=[value for _, value in normalized_pair],
                                        text=[value for _, value in normalized_pair])])
                fig.update_layout(title_text=header,
                                  plot_bgcolor='#fffaf4')
                fig.show()

        else:
            if plot == 'table':
                allexplainers = [x.keys() for x in self.averaged_metrics.values()]
                allmetrics = sorted(list(set([item for sublist in allexplainers for item in sublist])))

                headline = ['Metric']
                values = []

                for name, metrics in self.averaged_metrics.items():
                    headline += [name]
                    explainer_values = [fill_in_value(metrics, metric) for metric in allmetrics]
                    values.append(explainer_values)

                fig = go.Figure(data=[go.Table(
                    header=dict(values=headline,
                                line_color='#bfbfbf',
                                fill_color='#e0e5df',
                                align='left'),
                    cells=dict(values=[allmetrics] + values,  # 2nd column
                               line_color='#bfbfbf',
                               fill_color='#e0e5df',
                               align='left'))
                ])
                fig.update_layout(title_text=f'Metrics from all explainers')
                fig.show()

            elif plot == 'bar':
                normalized2_metrics = [(name, sorted(list(zip(metrics.keys(), metrics.values())), key=lambda tup: tup[0]))
                                       for name, metrics in normalize2(self.averaged_metrics).items()]

                fig = go.Figure(data=[go.Bar(name=name,
                                             x=[metric for metric, _ in normalized_pair],
                                             y=[value for _, value in normalized_pair],
                                             text=[value for _, value in normalized_pair])
                                      for (name, normalized_pair) in normalized2_metrics])
                fig.update_layout(barmode='group', title_text=f'Metric Comparison',
                                  plot_bgcolor='#ececea')
                fig.show()

