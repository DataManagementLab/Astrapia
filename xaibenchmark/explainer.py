

def metric(fn):
    """Decorator for tagging metrics.
    
    Metrics can be used to compare different explainers.
    """
    def wrapper(*args):
        result = fn(*args)
        if result is None:
            return float('nan')
        return result
    wrapper.tag = 'metric'
    return wrapper

def utility(fn):
    """Decorator for tagging utility functions.

    Utility functions can be used to infer different metrics automatically.   
    """
    # mark the method as something that can be used to infer metrics
    def wrapper(*args):
        result = fn(*args)
        if result is None:
            return float('nan')
        return result
    wrapper.tag = 'utility'
    return wrapper


class Explainer:
    """Base class for wrapping and comparing Explainers
    
    Add metrics using the @metric decorator.
    Add utility functions using the @utility decorator.

    When using the following predefined metrics and utilities, the library can infer other metrics by calling 
        your_explainer.infer_metrics()

    Metrics:
    - coverage(self)
    - 

    Utilities:
    - distance(self, x, y)
    - get_neighborhood_instances(self)
    -
    """
    
    def __init__(self):
        raise NotImplementedError
        
    def metrics(self):
        return [x for x in dir(self) if getattr(getattr(self, x), 'tag', None) == 'metric']
    
    def infer_metrics(self):
        
        # This should be set somewhere else
        transfer_graph = [
            ({'coverage'}, 'inverse_coverage', metric(lambda : 1 / self.coverage())),
            ({'distance', 'get_explained_instance', 'get_neighborhood_instances'}, 'furthest_distance', metric(lambda : max(0, 0, *[self.distance(self.get_explained_instance(), i) for i in self.get_neighborhood_instances()]))),
            ({'area', 'get_training_data'}, 'area_hc_normalised', metric(lambda : self.area()**(1/self.get_training_data().shape[1]))),
            ({'area', 'get_test_data'}, 'area_hc_normalised', metric(lambda : self.area()**(1/self.get_test_data().shape[1]))),
        ]

        mu_identifiers = {x for x in dir(self) if getattr(getattr(self, x), 'tag', None) in ['metric', 'utility']}
        
        old_mu_identifiers = {}
        new_mu_identifiers = mu_identifiers
        while (new_mu_identifiers != old_mu_identifiers):
            for transition in transfer_graph:
                if transition[0] <= new_mu_identifiers:
                    setattr(self, transition[1], transition[2])
                    
            old_mu_identifiers = new_mu_identifiers
            new_mu_identifiers = {x for x in dir(self) if getattr(getattr(self, x), 'tag', None) == 'metric'}
            
        print('inferred metrics:', new_mu_identifiers)
        
    def report(self):
        
        all_mu_identifier_references = {(x, getattr(self, x)) for x in dir(self) if getattr(getattr(self, x), 'tag', None) == 'metric'}
        implemented_mu_values = {(x, f()) for (x, f) in all_mu_identifier_references}
        return implemented_mu_values
                