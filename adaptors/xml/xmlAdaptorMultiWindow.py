#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Statistical Modelling of Human Trnaslation Processes
# Author: Samuel Läubli (slaubli@inf.ed.ac.uk)

from __future__ import division

import sys
from collections import defaultdict

from xmlAdaptor import *
from adaptors.observation import Observation

class XMLAdaptorMultiWindow1 ( AbstractXMLAdaptor ):
    '''
    Window-based extraction from CASMACAT XML. For each of the following events types,
    the number of occurrences in equally long windows of @param window_length (int) ms
    are aggregated:
        - number of keyDown alphanumeric, interpunctuation, ...
        - number of keyDown DEL, BACKSPACE
        - number of keyDown CTRL, ALT, WIN/MAC
        - number of keyDown arrow (left, right, up, down)
    '''
    
    def __init__ ( self, window_length=5000, use_duration=True ):
        """
        @param window_length (int): The length of the window in which counts will be
            aggregated, in milliseconds.
        @param use_duration (bool): If true, the observation value for all event types
            that have a duration attribute will be the cumulated duration of all obser-
            vations per window; normal counts are used otherwise.
        """
        events = ['keyDown']
        events += ['startSession', 'stopSession'] # as we consider subsessions in the logs
        self._use_duration = use_duration
        AbstractXMLAdaptor.__init__ ( self, events=events, parametrisation='window-based', window_length=window_length )
        # constants
        self._CONTROL_KEY_CODES = [17, 18] # ctrl, alt
        self._DELETE_KEY_CODES = [8, 46] # backspace, delete
        self._NAVIGATION_KEY_CODES = [33, 34, 35, 36, 37, 38, 39, 40] # page up, page down, end, home, left arrow, up arrow, right arrow, down arrow
        # features
        self._feature_names = ['keyDownNormal', 'keyDownDel', 'keyDownCtrl', 'keyDownNav']
    
    def convert ( self, xml_filepath ):
        '''
        Converts an XML file at @param xml_filepath into a list of observations
        of type Observation.
        '''
        self._subsessions = [] # will finally store one ObservationSession from each subsession (<startSession> to <stopSession>)
        self._observations = defaultdict(list) # (start, event, value) tuples (value) for subsessions (key)
        self._xml_filepath = xml_filepath
        with open(xml_filepath, 'r') as xml_file:
            context = etree.iterparse(xml_file, events=('end',), tag=self._events, recover=True)
            self._fast_iter(context)
        self._observations = self._getObservationSequence() # combine observation sequences from subsessions into a single observation sequence
        return self._observations
    
    def _processNode ( self, node ):
        timestamp = int( node.get('time') )
        if node.tag == 'startSession':
            # set timestamp for start of subsession
            self._subsessions.append(timestamp)
        elif node.tag == 'stopSession':
            # set timestamp for end of subsession
            self._subsessions[-1] = (self._subsessions[-1], timestamp)
        else:
            # determine observation type
            obs_type = self._getEventType(node)
            # determine event value; True = increase count, int = duration in ms
            if 'duration' in node.attrib:
                obs_value = int( node.get('duration') )
            else:
                obs_value = True
            # store observation
            subsession_index = len(self._subsessions)-1
            if obs_type:
                self._observations[subsession_index].append( (timestamp, obs_type, obs_value) )
    
    def _getEventType ( self, node ):
        if node.tag == 'keyDown':
            key_code = int( node.get('which') )
            if key_code in self._CONTROL_KEY_CODES:
                # control key event
                return 'keyDownCtrl'
            elif key_code in self._DELETE_KEY_CODES:
                # delete key event
                return 'keyDownDel'
            elif key_code in self._NAVIGATION_KEY_CODES:
                # navigation key event
                return 'keyDownNav'
            else:
                # alphanumeric, interpunctuation, or any other key event
                return 'keyDownNormal'
    
    def _getObservationSequence ( self ):
        """
        Creates an ObservationSequence for each subsession in self._subsessions and adds
        the observations in self._observations as Observation objects to them.
        """
        observation_sequence = ObservationSequence(feature_names=self._feature_names)
        for subsession_index, (subsession_start, subsession_end) in enumerate(self._subsessions):
            # create list for all Observations of this subsession
            observations = []
            start = subsession_start
            end = subsession_start + self._window_length
            # add an empty Observation for each window
            while end < subsession_end:
                observations.append( Observation(start=start, end=end, value=[0 for f in self._feature_names]) )
                start = end
                end += self._window_length
            observations.append( Observation(start=start, end=subsession_end, value=[0 for f in self._feature_names]) ) # for last window (may be shorter than self._window_length)
            # update counts/duration for each window (Observation) of this subsession
            for obs_start, obs_type, obs_value in self._observations[subsession_index]:
                window_index = (obs_start - subsession_start) // self._window_length
                window_value = observations[window_index].getValue()
                feature_index = self._feature_names.index(obs_type)
                if self._use_duration and obs_value != True:
                    window_value[feature_index] += obs_value # increase duration
                else:
                    window_value[feature_index] += 1 # increase count
                observations[window_index].setValue(window_value)
            # add windows (Observations) of current subsession to observation sequence
            for observation in observations:
                observation_sequence.addObservation(observation)
        return observation_sequence


class XMLAdaptorMultiWindow2 ( XMLAdaptorMultiWindow1 ):
    """
    The same as XMLAdaptorMultiWindow1, but with the following additional event types:
        - number of mouse clicks (mouseDown event)
    """
     
    def __init__ ( self, window_length=5000, use_duration=True ):
        XMLAdaptorMultiWindow1.__init__( self, window_length=window_length, use_duration=use_duration )
        self._events += ['mouseDown']
        self._feature_names += ['mouseDown']
     
    def _getEventType ( self, node ):
        if node.tag == 'keyDown':
            key_code = int( node.get('which') )
            if key_code in self._CONTROL_KEY_CODES:
                # control key event
                return 'keyDownCtrl'
            elif key_code in self._DELETE_KEY_CODES:
                # delete key event
                return 'keyDownDel'
            elif key_code in self._NAVIGATION_KEY_CODES:
                # navigation key event
                return 'keyDownNav'
            else:
                # alphanumeric, interpunctuation, or any other key event
                return 'keyDownNormal'
        elif node.tag == 'mouseDown':
            # mouseDown event
            return 'mouseDown'
 
 
class XMLAdaptorMultiWindow3 ( XMLAdaptorMultiWindow2 ):
    """
    The same as XMLAdaptorMultiWindow2, but with the following additional event types:
        - count (or duration) of eye fixation on source text
        - count (or duration) of eye fixation on target text
    """
     
    def __init__ ( self, window_length=5000, use_duration=True ):
        XMLAdaptorMultiWindow2.__init__( self, window_length=window_length, use_duration=use_duration )
        self._events += ['fixation']
        self._feature_names += ['fixationSource', 'fixationTarget']
        self._SOURCE_WINDOW = 1
        self._TARGET_WINDOW = 2
     
    def _getEventType ( self, node ):
        # keyDown
        if node.tag == 'keyDown':
            key_code = int( node.get('which') )
            if key_code in self._CONTROL_KEY_CODES:
                # control key event
                return 'keyDownCtrl'
            elif key_code in self._DELETE_KEY_CODES:
                # delete key event
                return 'keyDownDel'
            elif key_code in self._NAVIGATION_KEY_CODES:
                # navigation key event
                return 'keyDownNav'
            else:
                # alphanumeric, interpunctuation, or any other key event
                return 'keyDownNormal'
        # mouseDown
        elif node.tag == 'mouseDown':
            # mouseDown event
            return 'mouseDown'
        # fixation
        elif node.tag == 'fixation':
            window = int( node.get('window') )
            if window == self._SOURCE_WINDOW:
                # eye fixation on source text
                return 'fixationSource'
            elif window == self._TARGET_WINDOW:
                # eye fixation on target window
                return 'fixationTarget'
            else:
                return None # e.g., if window == 0 (unresolved eye fixation?)