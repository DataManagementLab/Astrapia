import inspect
from functools import partial

import astrapia as ast

_transferlist = []  # global variable keeping track of loaded transfer functions


def add_transfer(f):
    """
    Add a transfer function to the list of all global transfer functions.

    :param f: The transfer function to add.
    """
    params = list(inspect.signature(f).parameters)  # use inspect to get the parameters of the function
    _transferlist.append((params, f.__name__, lambda obj: f(
        *[getattr(obj, req) for req in params])))  # add transfer function with dependencies


def use_transfer(obj):
    """
    Use the transfer functions on an object. Will add new attributes to the object.

    :param obj: The object to use the transfer functions on.
    """
    mu_identifiers = {x for x in dir(obj) if getattr(getattr(obj, x), 'tag', None) in ['metric', 'utility']}

    old_mu_identifiers = {}
    new_mu_identifiers = mu_identifiers
    while new_mu_identifiers != old_mu_identifiers:
        for transition in _transferlist:
            if set(transition[0]) <= new_mu_identifiers and transition[1] not in new_mu_identifiers:
                setattr(obj, transition[1], ast.metric(partial(transition[2], obj)))

        old_mu_identifiers = new_mu_identifiers
        new_mu_identifiers = {x for x in dir(obj) if getattr(getattr(obj, x), 'tag', None) == 'metric'}
