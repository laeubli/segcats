#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Statistical Modelling of Human Trnaslation Processes
# Author: Samuel Läubli (slaubli@inf.ed.ac.uk)

from shared import toFloat

class Observation ( object ):
    '''
    A simple class that holds the start timestamp, end timestamp, and the observed value
    of an observation.
    '''
    
    def __init__ ( self, start=None, end=None, value=None ):
        if start:
            self.setStart(start)
        if end:
            self.setEnd(end)
        if value:
            self.setValue(value)
    
    # setters
    def setStart ( self, start ):
        '''
        @param start: the start timestamp in miliseconds (int)
        '''
        assert isinstance(start, int)
        self._start = start
    
    def setEnd ( self, end ):
        '''
        @param start: the end timestamp in miliseconds (int)
        '''
        assert isinstance(end, int)
        self._end = end
    
    def setValue ( self, value ):
        '''
        The value of an observation must be a list with one item per feature,
        even if there is only one feature.
        '''
        assert isinstance(value, list)
        # make sure we use the correct data type for floats
        for i, feature_value in enumerate(value):
            if isinstance(feature_value, float):
                value[i] = toFloat(value[i])
        self._value = value
    
    # getters
    def getStart ( self ):
        return self._start

    def getEnd ( self ):
        return self._end
    
    def getValue ( self ):
        return self._value