#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Global test driver for Single Gaussian HMM
"""

import sys

import model, fileIO
from shared import *

states =            [ 'START',       'H1',              'H2',               'H3',               'END' ]
#initial_obs_probs = None
initial_obs_probs = [ (None,None),   (100.0, 131134.0), (250.0,131134.0),   (1000.0,131134.0),  (None,None) ]


#features, observations = fileIO.readObservations('example.obs')
#features, observations = fileIO.readObservations('test_data/observations/exampleSingleGaussianFixationDuration/P/*.obs')
features, observations = fileIO.readObservations('test_data/observations/exampleSingleGaussianFixationDuration/P/P01_P11.xml.obs')

myHMM = model.SingleGaussianHMM(
                        states=states, 
                        observation_sequences=observations, 
                        initial_observation_probabilities=initial_obs_probs, 
                        topology='fully-connected', 
                        training_iterations=5, 
                        verbose=True 
                        )

myHMM.visualisePDFHistory()

print
print
print "*** TESTS ***"
print

print myHMM._transition_probs
print myHMM._observation_means_variances
print myHMM.transitionProb(0,2)
print myHMM.observationProb(0, 260.0) # should be None (non-emitting)
print myHMM.observationProb(1, 260.0)
print myHMM.observationProb(4, 260.0) # should be None (non-emitting)

test_obs_seq = [ [31.0], [100.0], [200.0], ]

# FORWARD
#print "Forward probability tests"

print myHMM._forwardTrellis( test_obs_seq )
print myHMM.forwardProbability( test_obs_seq ) # forward probability of the whole observation sequence (=total observation probability)
print myHMM.forwardProbability( test_obs_seq, 0 ) # forward probabilities for all states at time 0
print myHMM.forwardProbability( test_obs_seq, 1, 'H3' ) # forward probability for state H3 at time 1
print myHMM.forwardProbability( observations[0] )

# BACKWARD
print "\nBackward probability tests"

print myHMM._backwardTrellis( test_obs_seq )
print myHMM.backwardProbability( test_obs_seq ) # backward probability of the whole observation sequence (=total observation probability)
print myHMM.backwardProbability( test_obs_seq, 0 ) # backward probabilities for all states at time 0
print myHMM.backwardProbability( test_obs_seq, 1, 'H3' ) # backward probability for state H3 at time 1

# VITERBI
print "\nViterbi tests"
print myHMM.viterbiProbability( test_obs_seq )

# BAUM-WELCH
#print "\nBaum-Welch tests"
#myHMM._reestimateParameters( observations )