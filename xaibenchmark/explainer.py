

def metric(fn):
    # mark the method as something that can be calculated as a metric
    def wrapper(*args):
        result = fn(*args)
        if result is None:
            return float('nan')
        return result
    wrapper.tag = 'metric'
    return wrapper

def utility(fn):
    # mark the method as something that can be used to infer metrics
    def wrapper(*args):
        result = fn(*args)
        if result is None:
            return float('nan')
        return result
    wrapper.tag = 'utility'
    return wrapper


class Explainer:
    
    
    def __init__(self):
        raise NotImplementedError
        
    def metrics(self):
                
        def checkImplementedMetric(f):
            return True or f() != None
        
        all_metrics_strings = [x for x in dir(self) if getattr(getattr(self, x), 'tag', None) == 'metric']
        all_metrics = [getattr(self, m) for m in all_metrics_strings]
        implemented_metrics = [metric for metric, metric_name in zip(all_metrics, all_metrics_strings) if checkImplementedMetric(metric)]
        
        implemented_metric_names = set([metric_name for metric, metric_name in zip(all_metrics, all_metrics_strings) if checkImplementedMetric(metric)])
        return implemented_metric_names
    
    def infer_metrics(self):
        
        def checkImplementedMetric(f):
            return True or f() != None
    
        
        all_mu_identifiers = [x for x in dir(self) if getattr(getattr(self, x), 'tag', None) in ['metric', 'utility']]
        all_mu_references = [getattr(self, m) for m in all_mu_identifiers]
        implemented_mu_references = {mu for mu in all_mu_references if checkImplementedMetric(mu)}
        implemented_mu_names = set([metric_name for metric, metric_name in zip(all_mu_references, all_mu_identifiers) if checkImplementedMetric(metric)])
        
        transfer_graph = [
            ({'coverage'}, 'inverse_coverage', metric(lambda : 1 / self.coverage())),
            ({'distance', 'get_explained_instance', 'get_neighborhood_instances'}, 'furthest_distance', metric(lambda : max(0, 0, *[self.distance(self.get_explained_instance(), i) for i in self.get_neighborhood_instances()]))),
        ]
        
        old_mu_identifiers = {}
        new_mu_identifiers = implemented_mu_names
        while (new_mu_identifiers != old_mu_identifiers):
            for transition in transfer_graph:
                if transition[0] <= new_mu_identifiers:
                    setattr(self, transition[1], transition[2])
                    
            old_mu_identifiers = new_mu_identifiers
            all_mu_identifiers = [x for x in dir(self) if getattr(getattr(self, x), 'tag', None) == 'metric']
            all_mu_references = [getattr(self, m) for m in all_mu_identifiers]
            new_mu_identifiers = {metric_name for metric, metric_name in zip(all_mu_references, all_mu_identifiers) if checkImplementedMetric(metric)}
            
        print('inferred metrics:', new_mu_identifiers)
        
    def report(self):
        
        def checkImplementedMetric(f):
            return True or f() != None
        
        all_mu_identifier_references = {(x, getattr(self, x)) for x in dir(self) if getattr(getattr(self, x), 'tag', None) == 'metric'}
        implemented_mu_values = {(x, f()) for (x, f) in all_mu_identifier_references if checkImplementedMetric(f)}
        return implemented_mu_values
                