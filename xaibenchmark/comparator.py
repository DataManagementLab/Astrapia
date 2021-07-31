from collections import defaultdict


class ExplainerComparator:

    def __init__(self):
        self.explainers = {}
        self.averaged_metrics = {}
        self.explanations = {}
        self.metrics = {}

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

    def print_metrics(self, explainer="", index=None):
        """
        Output metrics of the explanations on the console. Either averaged metrics or metrics from single explanations
        :param explainer: Optional, in case you only want metrics from one explainer
        :param index: Optional, in case you only want metrics from one explanation
        """
        if explainer != "":
            if index is not None:
                output = self.metrics[explainer][index]
                print("Metric values for explainer", explainer,
                      ", explanation created with the", index, "-th instance of the given data")
            else:
                output = self.averaged_metrics[explainer]
                print("Average metric values for the explainer", explainer)
            for metric, value in output.items():
                print("\t", metric, ":", value)

        else:
            for name, metrics in self.averaged_metrics.items():
                print("Average metric values for explainer", name, ":")
                for metric, value in metrics.items():
                    print("\t", metric, ":", value)
