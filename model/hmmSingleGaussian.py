#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines and initialises a hidden Markov model
Observations consist of one continuous feature; observation probabilities are  modelled 
by a single Gaussian.
IMPORTANT: All probabilities are natural logarithms
"""

from __future__ import division

import sys, math, pdf
from shared import *
from collections import defaultdict

class SingleGaussianHMM:
    
    def __init__ ( self, parameter_file=None, states=None, observation_sequences=None, topology='fully_connected' ):
        """
        Creates and initialises a hidden Markov model. If @param from_file is given,
        all parameters are loaded from file. Otherwise, the parameters are initialised
        according to @param states, @param features, @param observations, and @param
        topology.
        @param from_file (str): path to file with HMM parameter definitions
        @param states (list): the hidden states. Must include 'START' and 'END'
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
        self._observation_means_variances = [] # matrix: prob = matrix[state] = (mean, variance)
        # model parameters from previous iterations
        self._previous_observation_probs = []
        self._previous_transition_probs = []
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
            self._init_observation_probabilities (observation_sequences)
    
    def _init_state_probabilities ( self, topology ):
        """
        Initialises the state transition probability matrix according to the given @param topology.
        """
        if isinstance(topology, str):
            self._transition_probs = self._create_transition_matrix(topology)
        else:
            self._transition_probs = topology
    
    def _init_observation_probabilities ( self, observation_sequences ):
        """
        Initialises the PDF for each state by taking the global mean and variance found in
        the @param observation_sequences.
        """
        # pool all observations out of all observation sequences
        all_observations = []
        for observation_sequence in observation_sequences:
            for observation in observation_sequence:
                all_observations.append(observation[0]) # single-valued observations only have one feature
        # calculate global mean and variance
        global_mean = mean(all_observations)
        global_variance = variance(all_observations)
        # add global mean and variance to each state
        for state in self._states:
            self._observation_means_variances.append( (global_mean, global_variance) )

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
        Returns the probability of seing @param observation in @param state.
        @param state (str/int): the name or index of the state
        @param observation (list): the observation, consisting of 1..(len(self._features)) features
        @return float: the probability of seing @param observation in @param state
        """
        if isinstance(observation, list):
            observation = observation[0] # since we're dealing with single-featured observations here
        assert(isinstance(observation, float))
        try:
            mean, variance = self._observation_means_variances[state]
            stdev = math.sqrt(variance)
            # normal distribution (single Gaussian)
            prob = ( 1 / (stdev * math.sqrt(2*math.pi)) ) * math.exp(-( ( math.pow((observation-mean),2) / (2*variance) ) ))
            return log(prob)
        except TypeError:
            # if state name rather than state index is provided
            self.observationProb(self._states.index(state), observation)

    def _forwardTrellis ( self, observation_sequence ):
        """
        Implements the Forward algorithm in log space. Computes the probability of seeing the
        observations in @param observation_sequence from start (time 0) to time t, given that 
        we are in state s at time t.
        @param observation_sequence: a list of observations, each observation being a list of
            1..* feature values (one for each feature of this HMM)
        @return (list): a trellis storing the Forward probability for each state (1st dimension;
            index of self._states) and time (2nd dimension; index of @param observation_sequence)
        """
        # create trellis (dynamic programming matrix)
        trellis = [] #trellis[state][time]
        t_to = len(observation_sequence)-1
        # initialisation
        for i, state in enumerate(self._states):
            trellis.append( [ logproduct(self.transitionProb(0,i), self.observationProb(i, observation_sequence[0])) ] )
        # fill trellis
        for t in range(1, t_to+1): # for each time t
            for j, state in enumerate(self._states): # for each state j
                fp_j = None
                for i, state in enumerate(self._states): # for each previous state i
                    fp_j = logsum( fp_j, logproduct(trellis[i][t-1],self.transitionProb(i,j)) )
                trellis[j].append( logproduct(fp_j, self.observationProb(j,observation_sequence[t])) )
        return trellis      
    
    def forwardProbability ( self, observation_sequence, time_t=None, state_s=None ):
        """
        Returns the Forward probability of a whole or partial sequence up to time t, either
        for a specified state or all possible states (see @return). 
        @param observation_sequence: a list of observations, each observation being a list of
            1..* feature values (one for each feature of this HMM)
        @param t (int): the index of the observation in @param observation_sequence at time t
        @param s (str/int): the name (str) or index (int) of the state s
        @return: If t and s are given: The forward probability in time t and state s
                 If t is given and s=None: A list of forward probabilities, the order of which
                     corresponds to each state in self._states
                 If t=None and s=None: The Forward probability of the whole observation sequence
                     from START to END.
        """
        if time_t == None:
            if state_s != None:
                sys.exit("Error computing Forward probability: time t must be given when state s is specified.")
        # get Forward probability trellis
        fp = self._forwardTrellis(observation_sequence)
        # return result according to provided parameters
        last_t_index = len(observation_sequence)-1
        end_state_index = len(self._states)-1
        if state_s == None:
            if time_t==None:
                # calculate total Forward probability
                forward_prob = None
                for i, state in enumerate(self._states): # for each previous state i
                            forward_prob = logsum( forward_prob, logproduct(fp[i][last_t_index],self.transitionProb(i,end_state_index)) ) # t is still set from above! (= last time step)
                return forward_prob
            else:
                return [forward_prob[time_t] for forward_prob in fp]
        else:
            try:
                return fp[state_s][time_t]
            except TypeError:
                # if the name rather than the index of the state s is provided
                try:
                    return fp[self._states.index(state_s)][time_t]
                except:
                    if state_s in [0, end_state_index, 'START', 'END']:
                        sys.exit("Error computing Backward probability: START and END states have no Backward probability at time t.") 
    
    def _backwardTrellis ( self, observation_sequence ):
        """
        Implements the Backward algorithm in log space. Computes the probability of seeing the
        observations in @param observation_sequence from start (time 0) to time t, given that 
        we are in state s at time t.
        @param observation_sequence: a list of observations, each observation being a list of
            1..* feature values (one for each feature of this HMM)
        @return (list): a trellis storing the forward probability for each state (1st dimension;
            index of self._states) and time (2nd dimension; index of @param observation_sequence)
        """
        # prepare ranges
        t = 0 # start from very first observation
        len_obs = len(observation_sequence)
        state_range = range(1, len(self._states)-1)
        # create trellis (dynamic programming matrix)
        trellis = [] #trellis[state][time] NOTE: Time is reversed in this trellis for now!
        # initialisation
        trellis.append([None]*len_obs) # for START state
        for i in state_range:
            trellis.append( [ self.transitionProb(i,len(self._states)-1) ] ) # transition probability from each possible state to 'END' state
        trellis.append([None]*len_obs) # for END state
        # fill trellis
        for t_b in range(0, len_obs-1-t): # t_b is the index of the next backprob (B_t+1) to be retrieved
            t_o = len_obs-1-t_b # t_o is the index of the next observation (o_t+1) to be retrieved
            for i in state_range:
                bp_i = None
                for j in state_range:
                    # bp_i = sum: self._transitionProb(i,j) * self._observationProb(j, t_o) * bp[j][t]
                    bp_i = logsum( bp_i, logproduct( self.transitionProb(i,j), logproduct( self.observationProb(j, observation_sequence[t_o]), trellis[j][t_b] ) ) )
                trellis[i].append(bp_i)
        # reverse order to make time t correspond to index of 2nd dimension in trellis
        return [trellis[state][::-1] for state in range(len(self._states))]
    
    def backwardProbability ( self, observation_sequence, time_t=None, state_s=None ):
        """
        Returns the Backward probability of a whole or partial sequence from time t+1, either
        for a specified state or all possible states (see @return). 
        @param observation_sequence: a list of observations, each observation being a list of
            1..* feature values (one for each feature of this HMM)
        @param t (int): the index of the observation in @param observation_sequence at time t
        @param s (str/int): the name (str) or index (int) of the state s
        @return: If t and s are given: The Backward probability in time t and state s
                 If t is given and s=None: A list of Backward probabilities, the order of which
                     corresponds to each state in self._states
                 If t=None and s=None: The Backward probability of the whole observation sequence
                     from START to END.
        """
        if time_t == None:
            if state_s != None:
                sys.exit("Error computing Backward probability: time t must be given when state s is specified.")
        # get Forward probability trellis
        bp = self._backwardTrellis(observation_sequence)
        # return according to provided parameters
        if state_s == None:
            if time_t==None:
                # return the Backward probability for the whole observation sequence
                backward_prob = None
                for j, state in enumerate(self._states): # for each previous state j
                            backward_prob = logsum( backward_prob, logproduct( self.transitionProb(0,j), logproduct( self.observationProb(j, observation_sequence[0]), bp[j][0] ) ) )
                return backward_prob
            else:
                # return the backward probabilities for all states at time t
                return [None] + [backward_prob[time_t] for backward_prob in bp[1:-1]] + [None]
        else:
            try:
                return bp[state_s][time_t]
            except TypeError:
                # if the name rather than the index of the state s is provided
                try:
                    return bp[self._states.index(state_s)][time_t]
                except:
                    if state_s in [0, len(self._states)-1, 'START', 'END']:
                        sys.exit("Error computing Backward probability: START and END states have no Backward probability at time t.")
    
    def viterbiProbability ( self, observation_sequence ):
        """
        Implements the Viterbi algorithm in log space. Finds the most probable sequence of
        states given an @param observation_sequence and this HMM's transition and observation
        parameters.
        @param observation_sequence: a list of observations, each observation being a list of
            1..* feature values (one for each feature of this HMM)
        @return (tuple): the most probable state sequence, encoded as a list of indices
            corresponding to self._states, as well as its score (log probability).
        """
        # initialise trellis
        trellis = [] # trellis[state][time] = (viterbi_prob, backpointer)
        for i, state in enumerate(self._states):
            first_viterbi = logproduct( self.transitionProb(0,i), self.observationProb(i, observation_sequence[0]) )
            first_backpointer = 0 # Index of START state
            trellis.append([(first_viterbi,first_backpointer)])
        # recursion
        for t, observation in enumerate(observation_sequence[1:], 1): # for every time t > 0
            # find maximum Viterbi probability at t-1
            max = (None,None)
            for i, state in enumerate(self._states):
                if trellis[i][t-1][0] > max[0]:
                    max = ( trellis[i][t-1][0], i )
            for j, state in enumerate(self._states):
                backpointer = max[1]
                viterbi = logproduct( max[0], logproduct( self.transitionProb(backpointer,j), self.observationProb(j, observation) ) )
                trellis[j].append( (viterbi, backpointer) )
        # termination
        # find maximum Viterbi probability at T
        max = (None,None)
        end_state_index = len(self._states)-1
        for i, state in enumerate(self._states):
            if trellis[i][t][0] > max[0]:
                max = ( trellis[i][t][0], i )
        # compute probability
        final_probability = logproduct( max[0], self.transitionProb(max[1],end_state_index) )
        # retrieve most probable state sequence via backpointers
        most_probable_state_sequence = [ end_state_index, max[1] ]
        previous_backpointer = max[1]
        i = len(observation_sequence)-1
        while i >= 0:
            previous_backpointer = trellis[previous_backpointer][i][1]
            most_probable_state_sequence.append(previous_backpointer)
            i -= 1
        return most_probable_state_sequence[::-1], final_probability # [::-1] reverses state sequence
        
    def _reestimateParameters ( self, observation_sequences ):
        """
        Updates the transition and observation probabilities for this HMM through one Baum-Welch
        (EM) training iteration on @param observation_sequences. This process can be iterated
        until the desired precision is reached.
        @param observation_sequence: a list of observations to learn the new parameters from,
            each observation being a list of 1..* feature values (one for each feature of this HMM)
        @return (TODO): TODO (information gain?)
        """
        # EXPECTATION STEP
        xi = [] # expected transitions at time t from state j to i
        gamma = [ ] # expected observations at time t in state j
        for k, observation_sequence in enumerate(observation_sequences):
            # get forward and backward probabilities for all states and times in observation
            forward_trellis = self._forwardTrellis(observation_sequence)
            backward_trellis = self._backwardTrellis(observation_sequence)
            # compute state occupation and transition probabilities
            xi.append( self._estimateXi(observation_sequence, forward_trellis, backward_trellis) )
            gamma.append( self._estimateGamma(observation_sequence, forward_trellis, backward_trellis) )
        # MAXIMISATION
        self._reestimateTransitionProbs(observation_sequences, xi, gamma)
        self._reestimateObservationProbs(observation_sequences, gamma)
    
    def _estimateXi ( self, observation_sequence, forward_trellis, backward_trellis ):
        """
        Estimates the probability of being in state s_i at time t and state s_j at time t+1, 
        given the current model parameters  and an @param observation_sequence.
        @param observation_sequence: a list of observations to learn the new parameters from,
            each observation being a list of 1..* feature values (one for each feature of this HMM)
        @return (list): A 3-dimensional array containing the probability of being in state s_i at 
            time t and state s_j at time t+1 for all states in self._states. The probabilities are
            stored as follows: array[t][i][j] = float.
        """
        xi = []
        for t, observation in enumerate(observation_sequence[:-1]): # only up to T-1 for i and T for j; how to deal with END state?
            denominator = None
            xi.append([])
            for i, state_i in enumerate(self._states):
                xi[t].append([])
                for j, state_j in enumerate(self._states):
                    xi_t_i_j = logproduct( forward_trellis[i][t],                              logproduct( \
                                           self.transitionProb(i,j),                           logproduct( \
                                           self.observationProb(j,observation_sequence[t+1]),  \
                                           backward_trellis[j][t+1] ) ) )
                    xi[t][i].append( xi_t_i_j )
                    denominator = logsum( denominator, xi_t_i_j ) # the denominator is the sum of all xi values
            for i, xi_t_i in enumerate(xi[t]):
                for j, xi_t_i_j in enumerate(xi_t_i):
                    xi[t][i][j] = logproduct( xi_t_i_j, -denominator ) # division (normalisation) in log space
        return xi
    
    def _estimateGamma ( self, observation_sequence, forward_trellis, backward_trellis ):
        """
        # TODO: This could be merged into one iteration over all t per observation sequence with
            self._extimateXi.
        ###
        Estimates the probability of being in state s_i at time t given the current model parameters
        and an @param observation sequence.
        @param observation_sequence: a list of observations to learn the new parameters from,
            each observation being a list of 1..* feature values (one for each feature of this HMM)
        @return (list): A 2-dimensional array containing the probability of being in state s_i at 
            time t for all states in self._states. The probabilities are stored as follows: 
            array[t][i] = float.
        """
        gamma = []
        for t, observation in enumerate(observation_sequence):
            denominator = None
            gamma.append([])
            for i, state in enumerate(self._states):
                gamma_t_i = logproduct( forward_trellis[i][t], backward_trellis[i][t] )
                gamma[t].append( gamma_t_i )
                denominator = logsum( denominator, gamma_t_i ) # the denominator is the sum of all gamma values
            for i, gamma_t in enumerate(gamma[t]):
                gamma[t][i] = logproduct( gamma_t, -denominator ) # division (normalisation) in log space
        return gamma
    
    def _reestimateTransitionProbs ( self, observation_sequences, xi, gamma ):
        """
        Estimates new transition probabilities for this HMM. The old transition probabilities
            are stored in self._previous_transition_probs.
        @param observation_sequence: a list of observations to learn the new parameters from,
            each observation being a list of 1..* feature values (one for each feature of this HMM)
        @param xi: the Xi values estimated for @param observation_sequences through self._estimateXi
        @param gamma: the Gamma values estimated for @param observation_sequences through
            self._estimateGamma
        """
        # implement reestimation formula
        def reestimateTransitionProb ( state_from, state_to ):
            """
            Reestimates the transition probability from @param state_from to @param state_to.
            @param state_from (int), state_to (int): the indices of the from and to state, respectively
            @return (float): the new transition probability from @param state_from to @param state_to
            """
            numerator = None
            denominator = None
            for k, observation_sequence in enumerate(observation_sequences):                
                k_prob = 1 / self.forwardProbability(observation_sequence) # EV. TODO: Only until T, not END (as long as END state probs are not fixed)
                xi_sum = None
                gamma_sum = None
                for t, observation in enumerate(observation_sequence[:-1]):
                    xi_sum =    logsum( xi_sum, xi[k][t][state_from][state_to] )
                    gamma_sum = logsum( gamma_sum, gamma[k][t][state_from] )
                numerator =   logsum ( numerator, logproduct( k_prob, xi_sum ) )
                denominator = logsum ( denominator, logproduct( k_prob, gamma_sum ) )
            if denominator == None:
                #print "Zero-probability for new transition probability from %s to %s" % (state_from, state_to)
                return None # -None raises a TypeError and cannot be handled by shared.logproduct()
            else:
                return logproduct( numerator, -denominator )
        # iterate over all possible state transitions
        new_transition_probs = []
        for i, state_i in enumerate(self._states):
            new_transition_probs.append([])
            for j, state_j in enumerate(self._states):
                new_transition_probs[i].append( reestimateTransitionProb(i,j) )
                print "%s -> %s %s" % (state_i, state_j, new_transition_probs[i][j]) 
                
    
    def _reestimateObservationProbs ( self, observation_sequences, gamma ):
        """
        Estimates new observation probabilities for this HMM. The old variances and gaussians are 
            stored in self._previous_transition_probs.
        @param observation_sequence: a list of observations to learn the new parameters from,
            each observation being a list of 1..* feature values (one for each feature of this HMM)
        @param gamma: the Gamma values estimated for @param observation_sequences through self._estimateGamma
        """
        # implement reestimation formula
        def reestimateMeanVariance ( state ):
            """
            Reestimates the mean and variance for @param state.
            @param state (int): the state's index
            @return (tuple): the new mean (float) and variance (float) for @param state
            """
            numerator_mean = None
            numerator_variance = None
            denominator = None
            for k, observation_sequence in enumerate(observation_sequences):                
                k_prob = 1 / self.forwardProbability(observation_sequence) # EV. TODO: Only until T, not END (as long as END state probs are not fixed)
                numerator_mean_sum = None
                numerator_variance_sum = None
                denominator_sum = None
                for t, observation in enumerate(observation_sequence[:-1]):
                    gamma_k_t_i = gamma[k][t][i]
                    current_mean_i = self._observation_means_variances[i][0]
                    numerator_mean_sum =     logsum( numerator_mean_sum, logproduct( gamma_k_t_i, log(observation[0]) ) )
                    numerator_variance_sum = logsum( numerator_variance_sum, logproduct( gamma_k_t_i, log(math.pow((observation[0]-current_mean_i),2)) ) )
                    denominator_sum =        logsum( denominator_sum, gamma_k_t_i )
                numerator_mean =      logsum ( numerator_mean, logproduct( k_prob, numerator_mean_sum ) )
                numerator_variance =  logsum ( numerator_variance, logproduct( k_prob, numerator_variance_sum ) )
                denominator =         logsum ( denominator, logproduct( k_prob, denominator_sum ) )
            if denominator == None:
                #print "Zero-probability for new transition probability from %s to %s" % (state_from, state_to)
                return None # -None raises a TypeError and cannot be handled by shared.logproduct()
            else:
                log_mean = logproduct( numerator_mean, -denominator )
                log_variance = logproduct( numerator_variance, -denominator ) 
                return exp(log_mean), exp(log_variance)
        # iterate over all states
        new_observation_means_variances = []
        for i, state in enumerate(self._states):
            new_observation_means_variances.append( reestimateMeanVariance(i) )
            print new_observation_means_variances[i]
            #print "%s: mean=%s, variance=%s" % (state, new_observation_means_variances[i][0], new_observation_means_variances[i][1])
            
        