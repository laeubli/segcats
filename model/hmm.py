#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines and initialises a hidden Markov model
"""

from __future__ import division

import sys, pdf
from collections import defaultdict

class HMM:
    
    def __init__ ( self, parameter_file=None, states=None, features=None, observation_sequences=None, topology='fully_connected' ):
        """
        Creates and initialises a hidden Markov model. If @param from_file is given,
        all parameters are loaded from file. Otherwise, the parameters are initialised
        according to @param states, @param features, @param observations, and @param
        topology.
        @param from_file (str): path to file with HMM parameter definitions
        @param states (list): the hidden states. Must include 'START' and 'END'
        @param features (list): the feature streams (observations); 1..n of
            'D' denotes a discrete feature (string vocabulary)
                Example: Hot, Cold, ...
            'G' denotes a continuous feature (float) to be modelled with a Gaussian (single Gaussian PDF)
                Example: 83.8, 289.32, ...
            'W' denotes a continuous feature (float) to be modelled with a Weibull distribution (Weibull PDF)
                Example: 83.8, 289.32, ...
        @param observation_sequences (list): 1..* sequences of lists, with each list representing one
            observation as 1..n features (n == len(@param features))
        @param topology: the HMM topology, i.e., how the hidden states are connected; 1 of
            'fully-connected' (ergodic) means that each state is connected to each other state,
                including itself (self-loop)
            'left-to-right' (Bakis) means that each state n is connected only to itself (self-loop) 
                and state n+1 (START and END don't have a self-loop)
            ALTERNATIVELY a transition probability matrix (list of lists) can be provided
                to define a custom HMM topology.
        """
        self._states = []
        self._features = []
        self._transition_probs = [] # matrix: prob = matrix[from-state_index][to-state_index]
        if parameter_file != None:
            # load parameters from file
            self._initialise_from_file(parameter_file)
        else:
            # initialise states
            if 'START' not in states or 'END' not in states:
                sys.exit("Cannot construct HMM. States must include both START and END.")
            self._states = states
            # initialise transition probabilities
            if isinstance(topology, str):
                self._transition_probs = self._create_transition_matrix(topology)
            else:
                self._transition_probs = topology
            # initialise observation probabilities
            observed_values = []
            for feature_type in features:
                 if feature_type == 'D':
                     observed_values.append(defaultdict(int))
                 else:
                     observed_values.append([])
            # iterate over observations
            for observation_sequence in observation_sequences:
                for observation in observation_sequence:
                    assert len(observation) == len(features)
                    for i, feature in enumerate(observation):
                        if features[i] == 'D':
                            observed_values[i][str(feature)] += 1 # increase type count for discrete feature
                        else:
                            observed_values[i].append(float(feature)) # add observation for continuous feature
            # initialise features
            for i, feature_type in enumerate(features):
                if feature_type == 'D':
                    self._features.append( pdf.DiscreteFeature (self._states, observed_values[i], plotInitialObservationProbs=True))
                    #print self._features[i].getStateProbabilities('ONE')
                elif feature_type == 'G':
                    self._features.append( pdf.GaussianFeature (self._states, observed_values[i], plotInitialObservationProbs=True))
                    #print self._features[i].getStateProbabilities(1.0)
                elif feature_type == 'W':
                    self._features.append( pdf.WeibullFeature  (self._states, observed_values[i], plotInitialObservationProbs=True))
                else:
                    sys.exit("Cannot construct HMM. Invalid feature type provided; only 'D' (discrete), 'G' (Gaussian), and 'W' (Weibull) are valid.")

    def _create_transition_matrix ( self, topology ):
        """
        Creates an initial matrix of transition probabilities, according to a default topology.
        @param topology: the HMM topology, i.e., how the hidden states are connected; 1 of
            'fully-connected' (ergodic) means that each state is connected to each other state,
                including itself (self-loop)
            'left-to-right' (Bakis) means that each state n is connected only to itself (self-loop) 
                and state n+1 (START and END don't have a self-loop)
        @return (list): The transition probability matrix (list of lists) with each index
            corresponding to the state indices in self._states. Example: the transition probability
            from state 0 to state 2 is stored in matrix[0][2].
        """
        matrix = []
        if topology not in ['fully-connected', 'left-to-right']:
            sys.exit("Error constructing HMM: %s is an unknown topology." % topology)
        else:
            if topology == 'fully-connected': # aka ergodic
                # calculate uniform probabilities
                total_states = len(self._states) # including START and END
                prob_from_start = 1 / (total_states - 2) # as a START has no self transition and no transition to END
                prob_from_any = 1 / (total_states - 1) # no transition back to START
                zero_prob = 0.0
                # initialise matrix
                for state in self._states:
                    row = []
                    if state == 'START':
                        for s in self._states:
                            if s in ['START', 'END']:
                                row.append(zero_prob)
                            else:
                                row.append(prob_from_start)
                    elif state == 'END':
                        row = [zero_prob]*total_states
                    else:
                        for s in self._states:
                            if s == 'START':
                                row.append(zero_prob) # no transition back to START
                            else:
                                row.append(prob_from_any)
                    matrix.append(row)
            elif topology == 'left-to-right': # special case of Bakis network (as in ASR)
                assert(self._states[0] == 'START')
                assert(self._states[-1] == 'END')
                total_states = len(self._states) # including START and END
                for i, state in enumerate(self._states):
                    if i == 0:
                        row = [0.0, 1.0] + (total_states-2)*[0.0] # probabilities from START
                    elif i == total_states-1:
                        row = [0.0]*total_states # probabilities from END
                    else:
                        row = i*[0.0] + [0.5,0.5] + (total_states-2-i)*[0.0]
                    matrix.append(row)
        return matrix
    