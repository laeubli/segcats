#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Global test driver for xml serialiser and reader classes
"""

from xml import *

# # SAVE XML
# 
# states = ['START', 'H1', 'H2', 'H3', 'END']
# features = ['G']
# transition_probabilities = [ [None,  0.33,   0.33,   0.33,   None],
#                              [None,  0.25,   0.25,   0.25,   0.25],
#                              [None,  0.25,   0.25,   0.25,   0.25],
#                              [None,  0.25,   0.25,   0.25,   0.25],
#                              [None,  None,   None,   None,   None] ]
# means_variances = [ (None,None), (100.0, 131134.0), (250.0,131134.0), (1000.0,131134.0), (None,None) ]
# 
# s = SingleGaussianHMM_XMLSerialiser( states, means_variances, transition_probabilities, comment="Test" )
# print s.getXML()
# 
# # LOAD XML
# 
# r = SingleGaussianHMM_XMLReader('/Users/sam/Desktop/model.xml')
# print r.getStates()
# print r.getTransitionProbabilities()
# print r.getMeansVariances()

reader = HMMReader('/Users/sam/Documents/ausbildung/uni/msc_ai/thesis/Models/MultivariateGaussianHMM/keyDown/A/P/3states/model_gmm.xml')
model = reader.getModel()
print model.gmms_[0].covars_