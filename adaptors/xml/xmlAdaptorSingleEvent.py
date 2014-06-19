#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Statistical Modelling of Human Trnaslation Processes
# Author: Samuel Läubli (slaubli@inf.ed.ac.uk)

from __future__ import division

import sys

from xmlAdaptor import *
from adaptors.observation import Observation

class XMLAdaptorSingleEventA ( AbstractXMLAdaptor ):
    '''
    Event-based extraction of the duration attribute of a single event type.
    Example: Extract the length of all fixation events.
    @pre: The event type in question must have a "duration" attribute.
    '''
    
    def __init__ ( self, event ):
        '''
        Consider exactly one event type @param event and use event-based extraction.
        '''
        assert isinstance(event, str)
        events = [event]
        AbstractXMLAdaptor.__init__ ( self, events=events, parametrisation='event-based' )
    
    def _processNode ( self, node ):
        start = int( node.get('time') )
        duration = int( node.get('duration') )
        end = start + duration
        # check if current event's start time overlaps with previous event's stop time
        try:
            if self._observations.getObservation(-1).getEnd() > start:
                sys.stderr.write("Warning: Event with ID %s in %s overlaps with previous event.\n" % (node.get('id'), self._xml_filepath) )
        except IndexError:
            pass
        # append new observation
        self._observations.addObservation( Observation(start=start, end=end, value=[float(duration)]) )


class XMLAdaptorSingleEventB ( AbstractXMLAdaptor ):
    '''
    Event-based extraction of the inter-event delay between any two successive
    events of a single event type. Note: At the beginning the session (and each
    subsession if applicable), the first event is not recorded since it obviously
    has no predecessor to be compared with.
    Example: Extract the delay to the previous keydown event for every keydown
    event.
    '''
    
    def __init__ ( self, event ):
        '''
        Consider exactly one event type @param event and use event-based extraction.
        '''
        assert isinstance(event, str)
        events = [event, 'startSession'] # we have to know about the beginning of the session (and subsessions)
        self._event_type = event
        self._start_timestamp_of_previous_event = None # will skip the first event in question since obviously no inter-event delay can be computed
        AbstractXMLAdaptor.__init__ ( self, events=events, parametrisation='event-based' )
        
    def _processNode ( self, node ):
        if node.tag == self._event_type:
            start = int( node.get('time') )
            if self._start_timestamp_of_previous_event: # this skips the first event in question
                delay = start - self._start_timestamp_of_previous_event
                try:
                    duration = int( node.get('duration') )
                    end = start + duration
                except: #if this event type has no duration
                    end = start
                self._observations.addObservation( Observation(start=start, end=end, value=[float(delay)]) )
            self._start_timestamp_of_previous_event = start
        elif node.tag == 'startSession':
            self._start_timestamp_of_previous_event = None # restart after session break


class XMLAdaptorSingleEventC ( AbstractXMLAdaptor ):
    '''
    Window-based extraction of the duration attribute of a single event type.
    Example: Extract the total duration of eye-fixations in windows of equal
    length. If an event's start and ent time falls between window boundaries,
    it's duration is split accordingly.
    @pre: The event type in question must have a "duration" attribute.
    @pre: The events in question cannot be overlapping in time.
    '''
    
    def __init__ ( self, event, window_length=5000 ):
        '''
        Consider exactly one event type @param event and use window-based extraction with
        windows of equal length @param window_length (in milliseconds).
        '''
        assert isinstance(event, str)
        assert isinstance(window_length, int)
        events = [event, 'startSession', 'stopSession'] # we have to figure out the beginning and end of the whole session
        self._event_type = event
        self._session_start = None
        self._session_stop = None
        self._window_length = window_length
        self._prev_node_end_timestamp = None # to check for overlapping events
        AbstractXMLAdaptor.__init__ ( self, events=events, parametrisation='window-based', window_length=window_length )
    
    def _processNode ( self, node ):
        
        if node.tag == self._event_type:
            # get attributes of current event
            start = int( node.get('time') )
            duration = float( node.get('duration') )
            end = start + duration
            # check if current event's start time overlaps with previous event's stop time
            if self._prev_node_end_timestamp and self._prev_node_end_timestamp > start:
                sys.stderr.write("Warning: Event with ID %s in %s overlaps with previous event.\n" % (node.get('id'), self._xml_filepath) )
            self._prev_node_end_timestamp = end
            # add observations, including empty observations for the time that has passed since the last event was recorded
            # add empty observations for time that has passed before this event
            while start > self._observations.getObservation(-1).getStart() + self._window_length:
                new_obs_start = self._observations.getObservation(-1).getStart() + self._window_length
                new_obs_end = new_obs_start + self._window_length
                self._observations.addObservation( Observation(start=new_obs_start, end=new_obs_end, value=[0.0]) )
            # add this event
            if end < self._observations.getObservation(-1).getStart() + self._window_length:
                # entire event falls into this window
                self._observations.getObservation(-1).setValue( [ self._observations.getObservation(-1).getValue()[0] + duration ] )
            else:
                # duration falls into multiple windows, so split it
                duration_for_current_window = (self._observations.getObservation(-1).getStart() + self._window_length) - start
                self._observations.getObservation(-1).setValue( [ self._observations.getObservation(-1).getValue()[0] + float(duration_for_current_window) ] )
                # duration that falls into subsequent windows
                duration_for_subsequent_windows = (start + duration) - (self._observations.getObservation(-1).getStart() + self._window_length)
                while  duration_for_subsequent_windows > self._window_length:
                    # windows that are fully spanned by the current observation
                    new_obs_start = self._observations.getObservation(-1).getStart() + self._window_length
                    new_obs_end = new_obs_start + self._window_length
                    new_obs_duration = self._window_length
                    self._observations.addObservation( Observation(start=new_obs_start, end=new_obs_end, value=[float(new_obs_duration)]) )
                    duration_for_subsequent_windows -= self._window_length
                # the window in which the current event's end falls into
                new_obs_start = self._observations.getObservation(-1).getStart() + self._window_length
                new_obs_end = new_obs_start + self._window_length
                new_obs_duration = duration_for_subsequent_windows
                self._observations.addObservation( Observation(start=new_obs_start, end=new_obs_end, value=[float(new_obs_duration)]) )
        elif node.tag == 'startSession':
            self._session_start = int( node.get('time') )
            self._observations.addObservation( Observation(start=self._session_start, end=self._session_start+self._window_length, value=[0.0]) ) # this is the first observation of the session
        elif node.tag == 'stopSession':
            self._session_stop = int( node.get('time') ) # note: we're only taking the last stopSession event, thus neglecting all pauses inside the session!
            while self._observations.getObservation(-1).getEnd() < self._session_stop:
                new_obs_start = new_obs_start = self._observations.getObservation(-1).getStart() + self._window_length
                new_obs_end = new_obs_start + self._window_length
                self._observations.addObservation( Observation(start=new_obs_start, end=new_obs_end, value=[0.0]) )
            # adjust length of last window if necessary
            if self._observations.getObservation(-1).getEnd() >= self._session_stop:
                self._observations.getObservation(-1).setEnd(self._session_stop)
            else:
                # append last empty session that fills the gap until the end
                new_obs_start = new_obs_start = self._observations.getObservation(-1).getStart() + self._window_length
                new_obs_end = self._session_stop
                self._observations.addObservation( Observation(start=new_obs_start, end=new_obs_end, value=[0.0]) )


class XMLAdaptorSingleEventD ( AbstractXMLAdaptor ):
    '''
    Window-based counting of all events of a single event type.
    Example: Count the number of fixation events in windows of equal length.
    The start time decides which window an event belongs to. The first window
    starts at the start time of the first event.
    '''
    
    def __init__ ( self, event, window_length=5000 ):
        '''
        Consider exactly one event type @param event and use window-based extraction with
        windows of equal length @param window_length (in milliseconds).
        '''
        assert isinstance(event, str)
        assert isinstance(window_length, int)
        events = [event, 'startSession', 'stopSession'] # we have to figure out the beginning and end of the whole session
        self._event_type = event
        self._session_start = None
        self._session_stop = None
        self._window_length = window_length
        AbstractXMLAdaptor.__init__ ( self, events=events, parametrisation='window-based', window_length=window_length )
    
    def _processNode ( self, node ):
        
        if node.tag == self._event_type:
            # get attributes of current event
            start = int( node.get('time') )
            # add observations, including empty observations for the time that has passed since the last event was recorded
            # add empty observations for time that has passed before this event
            while start > self._observations.getObservation(-1).getStart() + self._window_length:
                new_obs_start = self._observations.getObservation(-1).getStart() + self._window_length
                new_obs_end = new_obs_start + self._window_length
                self._observations.addObservation( Observation(start=new_obs_start, end=new_obs_end, value=[0.0]) )
            # add this event
            self._observations.getObservation(-1).setValue( [ self._observations.getObservation(-1).getValue()[0] + 1.0 ] )
        elif node.tag == 'startSession':
            self._session_start = int( node.get('time') )
            self._observations.addObservation( Observation(start=self._session_start, end=self._session_start+self._window_length, value=[0.0]) ) # this is the first observation of the session
        elif node.tag == 'stopSession':
            self._session_stop = int( node.get('time') ) # note: we're only taking the last stopSession event, thus neglecting all pauses inside the session!
            while self._observations.getObservation(-1).getEnd() < self._session_stop:
                new_obs_start = new_obs_start = self._observations.getObservation(-1).getStart() + self._window_length
                new_obs_end = new_obs_start + self._window_length
                self._observations.addObservation( Observation(start=new_obs_start, end=new_obs_end, value=[0.0]) )
            # adjust length of last window if necessary
            if self._observations.getObservation(-1).getEnd() >= self._session_stop:
                self._observations.getObservation(-1).setEnd(self._session_stop)
            else:
                # append last empty session that fills the gap until the end
                new_obs_start = new_obs_start = self._observations.getObservation(-1).getStart() + self._window_length
                new_obs_end = self._session_stop
                self._observations.addObservation( Observation(start=new_obs_start, end=new_obs_end, value=[0.0]) )