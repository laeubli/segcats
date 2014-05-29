#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test driver for hmm.py
"""

from hmm import *
import fileIO

states = ['START', 'H1', 'H2', 'H3', 'END']

#features, observations = fileIO.readObservations('example.obs')
features, observations = fileIO.readObservations('example2*.obs')

myHMM = HMM(states=states, features=features, observation_sequences=observations, topology='fully-connected')
print myHMM._transition_probs