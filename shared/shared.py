#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared functions
Note: log(), exp(), logsum(), and logproduct() closely follow Tobias P. 
Mann's tutorial “Numerically Stable Hidden Markov Model Implementation”.
"""

import math
import numpy as np

def toFloat ( numeral ):
    """
    Converts a number @param numeral in any representation to the currently
    used float data type in segcats. This allows to switch between datatypes
    with varying precision.
    """
    return np.float64(numeral)

def floatToStr ( float_nr ):
    """
    Returns a more precise string representation of @param float_nr (float) than str(float_nr).
    @return (str): the string representation of @param float_nr
    """ 
    return format(float_nr, '.64f')

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

def mean ( observations ):
    """
    Calculates the mean of 1..* floats.
    @param observations (list): list of floats
    @return (float): the mean of the values in @param observations
    """
    return sum(observations) / len(observations)

def variance ( observations ):
    """
    Calculates the variance of 1..* floats.
    @param observations (list): list of floats
    @return (float): the variance of the values in @param observations
    """
    m = mean(observations)
    nominator = 0.0
    for observation in observations:
        nominator += math.pow((observation-m), 2)
    return nominator / (len(observations)-1)