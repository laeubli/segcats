#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Global test driver for XML adaptors
"""

from xmlAdaptor import *

adaptor = XMLAdaptorSingleEventD('fixation')
observations = adaptor.convert('/Users/sam/Documents/ausbildung/uni/msc_ai/thesis/Data/TPR Raw/CFT13/Translog-II/P01_P11.xml')
# print
for observation in observations:
    print "%s\t%s\t%s" % (observation.getStart(), observation.getEnd(), observation.getValue())