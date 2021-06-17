

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