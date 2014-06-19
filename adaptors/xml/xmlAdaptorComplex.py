#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Statistical Modelling of Human Trnaslation Processes
# Author: Samuel Läubli (slaubli@inf.ed.ac.uk)

from __future__ import division

from xmlAdaptor import *
from adaptors.observation import Observation

class XMLAdaptorComplexA ( AbstractXMLAdaptor ):
    '''
    Event-based extraction from CASMACAT XML. For each of the following events types,
    the delay to the previous event of each event type (including the event type
    of the current event itself) is output. The event types are:
        - time since last keyDown alphanumeric, interpunctuation, ...
        - time since last keyDown DEL, BACKSPACE
        - time since last keyDown CTRL, ALT, WIN/MAC
        - time since last keyDown arrow (left, right, up, down)
    '''
    
    def __init__ ( self, add_duration=False ):
        """
        @param add_duration: If True, the end time of each observation will be set such
            that it coincides with the next observation's start time. This can be helpful
            for timeline visualisations.
        """
        events = ['keyDown']
        events += ['startSession']
        self._add_duration = add_duration
        self._prev_event_start_time = None
        AbstractXMLAdaptor.__init__ ( self, events=events, parametrisation='event-based' )
        # Constants
        self._CONTROL_KEY_CODES = [17, 18] # ctrl, alt
        self._DELETE_KEY_CODES = [8, 46] # backspace, delete
        self._NAVIGATION_KEY_CODES = [33, 34, 35, 36, 37, 38, 39, 40] # page up, page down, end, home, left arrow, up arrow, right arrow, down arrow
        # features
        self._feature_names = ['keyDownNormal', 'keyDownDel', 'keyDownCtrl', 'keyDownNav']
    
    def _processNode ( self, node ):
        start = int( node.get('time') )
        end = start
        if node.tag == 'startSession':
            self._time_elapsed = {} # timestamp of most recently seen event per type
            for feature in self._feature_names:
                self._time_elapsed[feature] = None
            # this resets the counter for each subsession
        else:
            # set end time of previous event
            if self._add_duration:
                try:
                    self._observations.getObservation(-1).setEnd(start)
                except IndexError:
                    pass
            # append new observation based on delays
            observation = []
            for feature in self._feature_names:
                try:
                    observation.append( start - self._time_elapsed[feature] )
                except TypeError:
                    observation.append( None ) # if this feature has not yet been seen
            # update delays
            if node.tag == 'keyDown':
                self._processKeyDown(node)
            self._observations.addObservation( Observation(start=start, end=end, value=observation) )
        self._prev_event_start_time = start
    
    def _processKeyDown ( self, node ):
        key_code = int( node.get('which') )
        timestamp = int( node.get('time') )
        if key_code in self._CONTROL_KEY_CODES:
            # control key event
            self._time_elapsed['keyDownCtrl'] = timestamp
        elif key_code in self._DELETE_KEY_CODES:
            # delete key event
            self._time_elapsed['keyDownDel'] = timestamp
        elif key_code in self._NAVIGATION_KEY_CODES:
            # navigation key event
            self._time_elapsed['keyDownNav'] = timestamp
        else:
            # alphanumeric, interpunctuation, or any other key event
            self._time_elapsed['keyDownNormal'] = timestamp
        