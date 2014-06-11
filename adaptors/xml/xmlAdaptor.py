#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Statistical Modelling of Human Trnaslation Processes
# Author: Samuel Läubli (slaubli@inf.ed.ac.uk)

from __future__ import division

from lxml import etree  # @UnresolvedImport

from adaptors.observation import * # imports the Observation class
from shared import toFloat

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
        with open(xml_filepath, 'r') as xml_file:
            context = etree.iterparse(xml_file, events=('end',), tag=self._events, recover=True)
            self._fast_iter(context)
        return self._observations


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
        self._observations.append( Observation(start=start, end=end, value=[toFloat(duration)]) )


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
                self._observations.append( Observation(start=start, end=end, value=[toFloat(delay)]) )
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
        AbstractXMLAdaptor.__init__ ( self, events=events, parametrisation='window-based', window_length=window_length )
    
    def _processNode ( self, node ):
        
        if node.tag == self._event_type:
            # get attributes of current event
            start = int( node.get('time') )
            duration = toFloat( node.get('duration') )
            end = start + duration
            # add observations, including empty observations for the time that has passed since the last event was recorded
            # add empty observations for time that has passed bevore this event
            while start > self._observations[-1].getStart() + self._window_length:
                new_obs_start = self._observations[-1].getStart() + self._window_length
                new_obs_end = new_obs_start + self._window_length
                self._observations.append( Observation(start=new_obs_start, end=new_obs_end, value=[0.0]) )
            # add this event
            if end < self._observations[-1].getStart() + self._window_length:
                # entire event falls into this window
                self._observations[-1].setValue( [ self._observations[-1].getValue()[0] + duration ] )
            else:
                # duration falls into multiple windows, so split it
                duration_for_current_window = (self._observations[-1].getStart() + self._window_length) - start
                self._observations[-1].setValue( [ self._observations[-1].getValue()[0] + toFloat(duration_for_current_window) ] )
                # duration that falls into subsequent windows
                duration_for_subsequent_windows = (start + duration) - (self._observations[-1].getStart() + self._window_length)
                while  duration_for_subsequent_windows > self._window_length:
                    # windows that are fully spanned by the current observation
                    new_obs_start = self._observations[-1].getStart() + self._window_length
                    new_obs_end = new_obs_start + self._window_length
                    new_obs_duration = self._window_length
                    self._observations.append( Observation(start=new_obs_start, end=new_obs_end, value=[toFloat(new_obs_duration)]) )
                    duration_for_subsequent_windows -= self._window_length
                # the window in which the current event's end falls into
                new_obs_start = self._observations[-1].getStart() + self._window_length
                new_obs_end = new_obs_start + self._window_length
                new_obs_duration = duration_for_subsequent_windows
                self._observations.append( Observation(start=new_obs_start, end=new_obs_end, value=[toFloat(new_obs_duration)]) )
        elif node.tag == 'startSession':
            self._session_start = int( node.get('time') )
            self._observations.append( Observation(start=self._session_start, end=self._session_start+self._window_length, value=[0.0]) ) # this is the first observation of the session
        elif node.tag == 'stopSession':
            self._session_stop = int( node.get('time') ) # note: we're only taking the last stopSession event, thus neglecting all pauses inside the session!
            while self._observations[-1].getEnd() < self._session_stop:
                new_obs_start = new_obs_start = self._observations[-1].getStart() + self._window_length
                new_obs_end = new_obs_start + self._window_length
                self._observations.append( Observation(start=new_obs_start, end=new_obs_end, value=[0.0]) )
            # adjust length of last window if necessary
            if self._observations[-1].getEnd() >= self._session_stop:
                self._observations[-1].setEnd(self._session_stop)
            else:
                # append last empty session that fills the gap until the end
                new_obs_start = new_obs_start = self._observations[-1].getStart() + self._window_length
                new_obs_end = self._session_stop
                self._observations.append( Observation(start=new_obs_start, end=new_obs_end, value=[0.0]) )


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
            # add empty observations for time that has passed bevore this event
            while start > self._observations[-1].getStart() + self._window_length:
                new_obs_start = self._observations[-1].getStart() + self._window_length
                new_obs_end = new_obs_start + self._window_length
                self._observations.append( Observation(start=new_obs_start, end=new_obs_end, value=[0.0]) )
            # add this event
            self._observations[-1].setValue( [ self._observations[-1].getValue()[0] + 1.0 ] )
        elif node.tag == 'startSession':
            self._session_start = int( node.get('time') )
            self._observations.append( Observation(start=self._session_start, end=self._session_start+self._window_length, value=[0.0]) ) # this is the first observation of the session
        elif node.tag == 'stopSession':
            self._session_stop = int( node.get('time') ) # note: we're only taking the last stopSession event, thus neglecting all pauses inside the session!
            while self._observations[-1].getEnd() < self._session_stop:
                new_obs_start = new_obs_start = self._observations[-1].getStart() + self._window_length
                new_obs_end = new_obs_start + self._window_length
                self._observations.append( Observation(start=new_obs_start, end=new_obs_end, value=[0.0]) )
            # adjust length of last window if necessary
            if self._observations[-1].getEnd() >= self._session_stop:
                self._observations[-1].setEnd(self._session_stop)
            else:
                # append last empty session that fills the gap until the end
                new_obs_start = new_obs_start = self._observations[-1].getStart() + self._window_length
                new_obs_end = self._session_stop
                self._observations.append( Observation(start=new_obs_start, end=new_obs_end, value=[0.0]) )