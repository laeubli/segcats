#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Statistical Modelling of Human Trnaslation Processes
# Author: Samuel Läubli (slaubli@inf.ed.ac.uk)

from __future__ import division

from xmlAdaptor import *
from adaptors.observation import Observation

class XMLAdaptorMultiDelay1 ( AbstractXMLAdaptor ):
    '''
    Event-based extraction from CASMACAT XML. For each of the following events types,
    the delay to the previous event of each other event type is output. The event types are:
        - time since last keyDown alphanumeric, interpunctuation, ...
        - time since last keyDown DEL, BACKSPACE
        - time since last keyDown CTRL, ALT, WIN/MAC
        - time since last keyDown arrow (left, right, up, down)
    The delay to the current event type is always 0.
    '''
    
    def __init__ ( self, add_duration=False, distinguish_keydowns=True ):
        """
        @param add_duration (bool): If True, the end time of each observation will be set
            such that it coincides with the next observation's start time. This can be
            helpful for timeline visualisations.
        @param distinguish_keydowns (bool): If True, keyDown events will be grouped into
            Normal, Del, Ctrl, and Nav (see above); if False, a single keyDown feature
            will be used.
        """
        events = ['keyDown']
        events += ['startSession']
        self._add_duration = add_duration
        self._distinguish_keydowns = distinguish_keydowns
        self._prev_event_start_time = None
        AbstractXMLAdaptor.__init__ ( self, events=events, parametrisation='event-based' )
        # Constants
        self._CONTROL_KEY_CODES = [17, 18] # ctrl, alt
        self._DELETE_KEY_CODES = [8, 46] # backspace, delete
        self._NAVIGATION_KEY_CODES = [33, 34, 35, 36, 37, 38, 39, 40] # page up, page down, end, home, left arrow, up arrow, right arrow, down arrow
        # features
        if self._distinguish_keydowns:
            self._feature_names = ['keyDownNormal', 'keyDownDel', 'keyDownCtrl', 'keyDownNav']
        else:
            self._feature_names = ['keyDown']
    
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
                self._setPreviousObservationEndTime(start)
            # update delay of this node's event type
            self._updateDelay(node)
            # append new observation based on delays
            self._addObservation(start,end)
        # update timestamp of previous event for next iteration
        self._prev_event_start_time = start
    
    def _setPreviousObservationEndTime ( self, end ):
        """
        Sets the previous observation's end time to @param end (int).
        """
        try:
            self._observations.getObservation(-1).setEnd(end)
        except IndexError:
            pass
    
    def _addObservation ( self, start, end ):
        """
        Adds a new observation with the delay from @param start (int) to all event types.
        """
        observation = []
        for feature in self._feature_names:
            try:
                observation.append( start - self._time_elapsed[feature] )
            except TypeError:
                observation.append( None ) # if this feature has not yet been seen
        self._observations.addObservation( Observation(start=start, end=end, value=observation) )
    
    def _updateDelay ( self, node ):
        if node.tag == 'keyDown':
            self._updateDelayKeyDown(node)
    
    def _updateDelayKeyDown ( self, node ):
        key_code = int( node.get('which') )
        timestamp = int( node.get('time') )
        if self._distinguish_keydowns:
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
        else:
            self._time_elapsed['keyDown'] = timestamp


class XMLAdaptorMultiDelay2 ( XMLAdaptorMultiDelay1 ):
    """
    The same as XMLAdaptorComplexA1, but with the following additional event types:
        - time since last mouse click (mouseDown event)
    """
    
    def __init__ ( self, add_duration=False, distinguish_keydowns=True ):
        XMLAdaptorMultiDelay1.__init__( self, add_duration, distinguish_keydowns )
        self._events += ['mouseDown']
        self._feature_names += ['mouseDown']
    
    def _updateDelay ( self, node ):
        if node.tag == 'keyDown':
            self._updateDelayKeyDown(node)
        elif node.tag == 'mouseDown':
            self._updateDelayMouseDown(node)
    
    def _updateDelayMouseDown ( self, node) :
        timestamp = int( node.get('time') )
        self._time_elapsed['mouseDown'] = timestamp


class XMLAdaptorMultiDelay3 ( XMLAdaptorMultiDelay2 ):
    """
    The same as XMLAdaptorComplexA2, but with the following additional event types:
        - time since last eye fixation on source text
        - time since last eye fixation on target text
    """
    
    def __init__ ( self, add_duration=False, distinguish_keydowns=True ):
        XMLAdaptorMultiDelay2.__init__( self, add_duration, distinguish_keydowns )
        self._events += ['fixation']
        self._feature_names += ['fixationSource', 'fixationTarget']
        self._SOURCE_WINDOW = 1
        self._TARGET_WINDOW = 2
    
    def _updateDelay ( self, node ):
        if node.tag == 'keyDown':
            self._updateDelayKeyDown(node)
        elif node.tag == 'mouseDown':
            self._updateDelayMouseDown(node)
        elif node.tag == 'fixation':
            self._updateDelayFixation(node)
    
    def _updateDelayFixation ( self, node ):
        timestamp = int( node.get('time') )
        try:
            window = int( node.get('window') )
            if window == self._SOURCE_WINDOW:
                # eye fixation on source text
                self._time_elapsed['fixationSource'] = timestamp
            elif window == self._TARGET_WINDOW:
                # eye fixation on target text
                self._time_elapsed['fixationTarget'] = timestamp
        except TypeError:
            # post CFT13 format does not store a window attribute anymore
            if 'source' in node.get('elementId'):
                # eye fixation on source text
                self._time_elapsed['fixationSource'] = timestamp
            elif 'editarea' in node.get('elementId'):
                # eye fixation on target text
                self._time_elapsed['fixationTarget'] = timestamp