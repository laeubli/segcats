#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Statistical Modelling of Human Trnaslation Processes
# Author: Samuel Läubli (slaubli@inf.ed.ac.uk)

class Observation ( object ):
    '''
    A simple class that holds the start timestamp, end timestamp, and the observed
    feature values of an observation.
    '''
    
    def __init__ ( self, start=None, end=None, value=None, state=None ):
        self.setStart(start)
        self.setEnd(end)
        self.setValue(value)
        self.setState(state)

    # setters
    def setStart ( self, start ):
        '''
        @param start: the start timestamp in miliseconds (int)
        '''
        assert isinstance(start, int) or start == None
        self._start = start
    
    def setEnd ( self, end ):
        '''
        @param start: the end timestamp in miliseconds (int)
        '''
        assert isinstance(end, int) or end == None
        self._end = end
    
    def setValue ( self, value ):
        '''
        The value of an observation must be a list with one item per feature,
        even if there is only one feature.
        '''
        assert isinstance(value, list) or value == None
        self._value = value
    
    def setState ( self, state ):
        '''
        @param state: the name of this observation's state
        '''
        assert isinstance(state, str) or state == None
        self._state = state
    
    # getters
    def getStart ( self ):
        return self._start

    def getEnd ( self ):
        return self._end
    
    def getValue ( self ):
        return self._value

    def getState ( self ):
        return self._state