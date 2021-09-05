import inspect
import xaibenchmark as xb
from functools import partial

_transferlist = [] # variable keeping track of loaded transfer functions
    
def add_transfer(f):
    params = list(inspect.signature(f).parameters)
    _transferlist.append((params, f.__name__, lambda obj:f(*[getattr(obj, req) for req in params])))
    
def use_transfer(obj):
    mu_identifiers = {x for x in dir(obj) if getattr(getattr(obj, x), 'tag', None) in ['metric', 'utility']}

    old_mu_identifiers = {}
    new_mu_identifiers = mu_identifiers
    while new_mu_identifiers != old_mu_identifiers:
        for transition in _transferlist:
            if set(transition[0]) <= new_mu_identifiers and transition[1] not in new_mu_identifiers:
                setattr(obj, transition[1], xb.metric(partial(transition[2], obj)))
                
        old_mu_identifiers = new_mu_identifiers
        new_mu_identifiers = {x for x in dir(obj) if getattr(getattr(obj, x), 'tag', None) == 'metric'}
