#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Provides file input/output functionality.
"""
import sys, glob
from shared import toFloat

def readObservations ( path ):
    """
    Reads 1..* observation sequences, one per file. Each observation file consists of:
        - a definition of the features (D, G, W) in the first line (<tab>-separated)
        - one observation per line, with its features separated by <tab>
    Note that all files need to have the same feature definitions.
    @param path (str): The path to the observation file(s). Accepts wildcards such
        as /foo/bar/*.obs.
    @return (tuple): a (features, observation_sequences) tuple; the values can be
        handed to the hmm.HMM class to initialise a hidden Markov model.
    """
    def validateFeature( name ):
        """
        Checks if @param name is a valid feature type
        """
        if name in ['D','G','W']:
            return True
        else:
            sys.exit("Error reading observation sequences: %s is not a valid feature type." % name)
            
    features = []
    observation_sequences = []
    
    for file_path in glob.glob(path):
        observation_sequence = []
        with open(file_path, 'r') as file:
            line_number = 0
            for line in file:
                if line != '' and not line.startswith('#'):
                    line = line.strip()
                    line = line.split('#')[0] # discard comments after any hashtag
                    values = [value.strip() for value in line.split('\t') if value != '']
                    if line_number == 0:
                        for feature in values:
                            validateFeature(feature)
                        if features == []:
                            features = values
                            features_initialised = True
                        else:
                            if features != values:
                                sys.exit("Error reading observation sequences: The feature definition in %s differs from the previously read files." % file_path)
                    else:
                        observation = []
                        for i, value in enumerate(values):
                            if features[i] == 'D':
                                observation.append(str(value)) # force conversion to string for discrete feature values
                            else:
                                observation.append(toFloat(value)) # force conversion to float for continuous feature values (Gaussian, Weibull)
                        observation_sequence.append(observation)
                    line_number +=1
        observation_sequences.append(observation_sequence)
    
    return features, observation_sequences
                