#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines and initialises a hidden Markov model
IMPORTANT: All probabilities are natural logarithms, normally derived through math.log()
"""

from __future__ import division

import sys, math, pdf
from shared import *
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
            self._init_state_probabilities(topology)
            # initialise features with observation probabilities (uniform)
            self._init_observation_probabilities (features, observation_sequences )
    
    def _init_state_probabilities ( self, topology ):
        """
        Initialises the state transition probability matrix according to the given @param topology.
        """
        if isinstance(topology, str):
            self._transition_probs = self._create_transition_matrix(topology)
        else:
            self._transition_probs = topology
    
    def _init_observation_probabilities ( self, features, observation_sequences ):
        """
        Initialises a feature per feature stream and the observation probabilities for each state
        and feature. The initial observation probabilities are the probability distribution
        observed in @param observation_sequences, i.e., it is uniform for all states per feature.
        """
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
                #print self._features[i].getProbability(1, 20.0)
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
                prob_from_start = log( 1 / (total_states - 2) ) # as a START has no self transition and no transition to END
                prob_from_any = log( 1 / (total_states - 1) ) # no transition back to START
                zero_prob = log(0) # special value (None)
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
                full_prob = log(1.0)
                half_prob = log(0.5)
                zero_prob = log(0) # special value (None)
                for i, state in enumerate(self._states):
                    if i == 0:
                        row = [zero_prob, full_prob] + (total_states-2)*[zero_prob] # probabilities from START
                    elif i == total_states-1:
                        row = [zero_prob]*total_states # probabilities from END
                    else:
                        row = i*[zero_prob] + [half_prob,half_prob] + (total_states-2-i)*[zero_prob]
                    matrix.append(row)
        return matrix
    
    def transitionProb ( self, from_state, to_state ):
        """
        Returns the probability of going from @param start_state to @param end_state.
        @param start_state (str/int): the name or index of the start state
        @param end_state (str/int): the name or index of the end state 
        @return: the probability of going from @param start_state to @param end_state
        """
        try:
            return self._transition_probs[from_state][to_state]
        except TypeError:
            if isinstance(from_state, str):
                from_state = self._transition_probs.index(from_state)
            if isinstance(to_state, str):
                from_state = self._transition_probs.index(to_state)
            return self.transtitionProb(from_state, to_state)
    
    def observationProb ( self, state, observation ):
        """
        Returns the probability of seing @param observation in @param state, given all feature
        streams.
        @param state (str/int): the name or index of the state
        @param observation (list): the observation, consisting of 1..(len(self._features)) features
        @return float: the probability of seing @param observation in @param state
        """
        try:
            observation_prob = 0.0
            for i, feature in enumerate(self._features):
                observation_prob += feature.getProbability(state, observation[i])
            return observation_prob
        except TypeError:
            return None # special case for zero probability
    
    def forwardProb ( self, observation_sequence ):
        """
        Implements the forward algorithm in log space.
        @param observation_sequence: a list of observations, each observation being a list of
            1..* feature values (one for each feature of this HMM).
        @return: The joint probability of seeing @param observation_sequence, given this HMM's
            transition and observation parameters.
        """
        # create trellis (dynamic programming matrix)
        fp = [] #fp[state][time]
        # initialisation
        for i, state in enumerate(self._states):
            fp.append( [ logproduct(self.transitionProb(0,i), self.observationProb(i, observation_sequence[0])) ] )
        # fill trellis
        for t in range(1, len(observation_sequence)): # for each time t
            for j, state in enumerate(self._states): # for each state j
                fp_j = None
                for i, state in enumerate(self._states): # for each previous state i
                    fp_j = logsum( fp_j, logproduct(fp[i][t-1],self.transitionProb(i,j)) )
                fp[j].append( logproduct(fp_j, self.observationProb(j,observation_sequence[t])) )
        # calculate final probability
        forward_prob = None
        end_state_index = len(self._states)-1
        for i, state in enumerate(self._states): # for each previous state i
                    forward_prob = logsum( forward_prob, logproduct(fp[i][t],self.transitionProb(i,end_state_index)) ) # t is still set from above! (= last time step)
        return forward_prob
            