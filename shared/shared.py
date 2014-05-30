#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared functions
"""

import math

def log( prob ):
    """
    Converts a normal probability (float) to a log probability (float).
    Zero probabilities are undefined in the log space (minus infinity); we
    model this value as None in this framework.
    """
    if not isinstance(prob, float):
        prob = float(prob)
    if prob == 0.0:
        return None
    else:
        return math.log(prob)

def exp( prob ):
    """
    Converts a log probability (float) back to a normal probability (float).
    Minus infinite log probabilities are modelled as None in this framework;
    we return 0 in this case.
    """
    if prob == None:
        return 0.0
    elif not isinstance(prob, float):
        prob = float(prob)
    return math.exp( prob )