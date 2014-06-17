#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Statistical Modelling of Human Trnaslation Processes
# Author: Samuel Läubli (slaubli@inf.ed.ac.uk)

from __future__ import division

import sys
from lxml import etree  # @UnresolvedImport

from adaptors.observation import * # imports the Observation class

'''
Converts a TranslogII-style XML document into an observation sequence to be used for
model training and sequence tagging with segcats.
---
NOTE: All adaptors implemented can handle pauses in recorded translation sessions; events
and windows are only counted/computed inside <startSession>...<endSession> blocks. For
window-based parametrisation, this means that the length of windows at the boundaries of
such blocks may vary, i.e., the last window may be shorter than the defined window length.
'''

class AbstractXMLAdaptor ( object ):
    '''
    Implements the general XML adaptor mechanism. 
    '''
    
    def __init__ ( self, events=None, parametrisation='event-based', window_length=5000 ):
        '''
        Initialises the adaptor.
        @param events (list): A list of event node names (str) to be considered.
            Example: ['keydown', 'fixation']
        @param parametrisation (str): The way how user activity data (UAD) in the XML
            will be parametrised; one of
            'event-based':  each event forms one observation
            'window-based': the UAD is split into windows of length @param window_length
            and events are accumulated in each window; that is, each window forms one
            observation
        @param window_length (int): The length of each window in milliseconds. Only
            effective if @param parametrisation == 'window-based'.
        '''
        self._events = events
        self._parametrisation = parametrisation
        self._window_length = window_length
    
    def _fast_iter ( self, context ):
        '''
        Quickly iterates over all elements of an xml tree (lxml).
        See http://www.ibm.com/developerworks/xml/library/x-hiperfparse/.
        '''
        for event, elem in context:
            self._processNode(elem)
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
        del context
    
    def _processNode ( self, node ):
        '''
        Processes each XML node specified in self._events (passed via self._fast_iter).
        This class should append Observation objects to self._observations and be redefined
        in subclasses that extend AbstractXMLAdaptor.
        '''
        pass
    
    def _aggregate_observations ( self, window_length ):
        '''
        Aggregates all observations in self._observations into new observations, such that
        all observations in a time window of length @param window_length (int, in milliseconds)
        form one new observation.
        This class should append Observation objects to self._observations and be redefined
        in subclasses that extend AbstractXMLAdaptor.
        '''
        # new_observations = []
        # ...
        # self._observations = new_observations
        pass
    
    def convert ( self, xml_filepath ):
        '''
        Converts an XML file at @param xml_filepath into a list of observations
        of type Observation.
        '''
        self._observations = [] # Observation objects will be added in self._processNode
        self._xml_filepath = xml_filepath
        with open(xml_filepath, 'r') as xml_file:
            context = etree.iterparse(xml_file, events=('end',), tag=self._events, recover=True)
            self._fast_iter(context)
        return self._observations