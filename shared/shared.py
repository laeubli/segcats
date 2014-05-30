#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared functions
Note: log(), exp(), logsum(), and logproduct() closely follow Tobias P. 
Mann's tutorial “Numerically Stable Hidden Markov Model Implementation”.
"""

import math

def log( prob ):
    """
    Converts a normal probability (float) to a log probability (float).
    Zero probabilities are undefined in the log space (minus infinity); we
    model this value as None in this framework.
    """
    try:
        return math.log(prob)
    except ValueError:
        return None # special case for 0

def exp( prob ):
    """
    Converts a log probability (float) back to a normal probability (float).
    Minus infinite log probabilities are modelled as None in this framework;
    we return 0 in this case.
    """
    try:
        return math.exp( prob )
    except TypeError:
        return 0.0 # special case for minus infinity
    

def logsum ( logprob1, logprob2 ):
    """
    Returns the sum of two log probabilities (as output by log()).
    """
    if logprob1 == None or logprob2 == None:
        if logprob1 == None:
            return logprob2
        else:
            return logprob1
    else:
        if logprob1 > logprob2:
            return logprob1 + log( 1 + math.exp(logprob2-logprob1) )
        else:
            return logprob2 + log( 1 + math.exp(logprob1-logprob2) )

def logproduct ( logprob1, logprob2 ):
    """
    Returns the product of two log probabilities (as output by log()).
    """
    try:
        return logprob1 + logprob2
    except TypeError:
        return None # if either logprob1 or logprob2 are None (i.e., minus infinity)