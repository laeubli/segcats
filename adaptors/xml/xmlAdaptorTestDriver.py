#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Global test driver for XML adaptors
"""
import sys

from adaptors.xml import *
from fileIO import *

#adaptor = XMLAdaptorSingleEventD('fixation')
#adaptor = XMLAdaptorMultiDelay3(add_duration=True)
adaptor = XMLAdaptorMultiWindow3(use_duration=False, distinguish_keydowns=False)
observation_sequence = adaptor.convert('/Users/sam/Documents/ausbildung/uni/msc_ai/thesis/Data/TPR Raw/CFT13/Translog-II/P07_P21.xml')
#observation_sequence = adaptor.convert('test_log.xml')

#print observation_sequence.getNumpyArray(replaceNone="TEST")[0]

for observation in observation_sequence.get():
    print "%s\t%s\t%s" % (observation.getStart(), observation.getEnd(), observation.getValue())

# serialise
observation_sequence.save('test_sequence.csv')
# load and compare
loaded_observation_sequence = ObservationSequence('test_sequence.csv')
for i, observation in enumerate(observation_sequence.get()):
    if observation.getStart() != loaded_observation_sequence.getObservation(i).getStart():
        sys.stderr.write('Warning: Saving/loading affected the start value of observation %s \n' % i)
    if observation.getEnd() != loaded_observation_sequence.getObservation(i).getEnd():
        sys.stderr.write('Warning: Saving/loading affected the end value of observation %s \n' % i)
    if observation.getValue() != loaded_observation_sequence.getObservation(i).getValue():
        sys.stderr.write('Warning: Saving/loading affected the feature values of observation %s \n' % i)