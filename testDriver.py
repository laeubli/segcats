#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Global test driver
"""

import model, fileIO
from shared import *

states = ['START', 'H1', 'H2', 'H3', 'END']

#features, observations = fileIO.readObservations('example.obs')
features, observations = fileIO.readObservations('test_data/observations/example2/*.obs')

myHMM = model.HMM(states=states, features=features, observation_sequences=observations, topology='fully-connected')
#print myHMM._transition_probs
#print myHMM.transitionProb(0,2)
#print myHMM.observationProb(2, ['keydown', 31.0, 'source'])

test_obs_seq = [ ['keydown', 31.0, 'source'], ['keydown', 100.0, 'source'], ['fixation', 200.0, 'target'], ]

# FORWARD
print "Forward probability tests"

print myHMM._forwardTrellis( test_obs_seq )
print myHMM.forwardProbability( test_obs_seq ) # forward probability of the whole observation sequence (=total observation probability)
print myHMM.forwardProbability( test_obs_seq, 0 ) # forward probabilities for all states at time 0
print myHMM.forwardProbability( test_obs_seq, 1, 'H3' ) # forward probability for state H3 at time 1

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
print "\nBaum-Welch tests"
myHMM._reestimateParameters( observations )