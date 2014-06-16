#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Global test driver for XML adaptors
"""
import sys

from xmlAdaptor import *
from fileIO import serialiseObservationSequence, readObservationSequence

adaptor = XMLAdaptorSingleEventC('fixation')
observation_sequence = adaptor.convert('/Users/sam/Documents/ausbildung/uni/msc_ai/thesis/Data/TPR Raw/CFT13/Translog-II/P01_P11.xml')
#observation_sequence = adaptor.convert('test_log.xml')

for observation in observation_sequence:
    print "%s\t%s\t%s" % (observation.getStart(), observation.getEnd(), observation.getValue())
# serialise
serialiseObservationSequence(observation_sequence, 'test_sequence.csv', ['value'])
# load and compare
loaded_observation_sequence = readObservationSequence('test_sequence.csv')
for i, observation in enumerate(observation_sequence):
    if observation.getStart() != loaded_observation_sequence[i].getStart():
        sys.stderr.write('Warning: Saving/loading affected the start value of observation %s \n' % i)
    if observation.getEnd() != loaded_observation_sequence[i].getEnd():
        sys.stderr.write('Warning: Saving/loading affected the end value of observation %s \n' % i)
    if observation.getValue() != loaded_observation_sequence[i].getValue():
        sys.stderr.write('Warning: Saving/loading affected the feature values of observation %s \n' % i)