

def metric(fn):
    # mark the method as something that requires view's class
    fn.tag = 'metric'
    return fn


class Explainer:
    
    
    def __init__(self):
        raise NotImplementedError
        
    @metric
    def area(self):
        raise NotImplementedError
        
    @metric
    def coverage(self):
        raise NotImplementedError
        
    def metrics(self):
                
        def checkImplemented(f):
            try:
                f()
            except NotImplementedError:
                return False
            return True
        
        all_metrics_strings = [x for x in dir(self) if getattr(getattr(self, x), 'tag', None) == 'metric']
        all_metrics = [getattr(self, m) for m in all_metrics_strings]
        implemented_metrics = [metric for metric, metric_name in zip(all_metrics, all_metrics_strings) if checkImplemented(metric)]
        
        implemented_metric_names = set([metric_name for metric, metric_name in zip(all_metrics, all_metrics_strings) if checkImplemented(metric)])
        return implemented_metric_names
    
    def infer_metrics(self):
        
        def checkImplemented(f):
            try:
                f()
            except NotImplementedError:
                return False
            return True
        
        all_metrics_strings = [x for x in dir(self) if getattr(getattr(self, x), 'tag', None) == 'metric']
        all_metrics = [getattr(self, m) for m in all_metrics_strings]
        implemented_metrics = {metric for metric, metric_name in zip(all_metrics, all_metrics_strings) if checkImplemented(metric)}
        implemented_metric_names = set([metric_name for metric, metric_name in zip(all_metrics, all_metrics_strings) if checkImplemented(metric)])
        
        transfer = [
            ({'coverage'}, 'inverse_coverage', lambda : 1 / self.coverage()),
        ]
        
        old_metrics = {}
        new_metrics = implemented_metric_names
        while (new_metrics != old_metrics):
            for transfer_list in transfer:
                if transfer_list[0] <= new_metrics:
                    setattr(self, transfer_list[1], metric(transfer_list[2]))
                    
            old_metrics = new_metrics
            all_metrics_strings = [x for x in dir(self) if getattr(getattr(self, x), 'tag', None) == 'metric']
            all_metrics = [getattr(self, m) for m in all_metrics_strings]
            new_metrics = {metric_name for metric, metric_name in zip(all_metrics, all_metrics_strings) if checkImplemented(metric)}
            
        print('inferred metrics:', new_metrics)
        
    def report(self):
        
        def checkImplemented(f):
            try:
                f()
            except NotImplementedError:
                return False
            return True
        
        all_metrics = {(x, getattr(self, x)) for x in dir(self) if getattr(getattr(self, x), 'tag', None) == 'metric'}
        implemented_metrics = {(x, f()) for (x, f) in all_metrics if checkImplemented(f)}
        return implemented_metrics
                