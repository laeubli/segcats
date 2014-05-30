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
print myHMM.forwardProb( [ ['keydown', 31.0, 'source'], ['keydown', 100.0, 'source'], ['fixation', 200.0, 'target'], ])