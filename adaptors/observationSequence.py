#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Statistical Modelling of Human Trnaslation Processes
# Author: Samuel Läubli (slaubli@inf.ed.ac.uk)

import sys
import numpy as np

from observation import Observation
from fileIO.fileIO import *

class ObservationSequence ( object ):
    '''
    Holds an ordered sequence of Observations.
    '''
    
    def __init__ ( self, file_path=None, feature_names=None ):
        """
        @param csv_filepath (str): If the path to a CSV file is given, its content is
            loaded upon initialisation.
        @param feature_names (list): The list of feature names for each feature value
            of this ObservationSequence's Observations.
        """
        self._observations = []
        self._observation_feature_names = [] # all CSV column rows, except for 'start', 'end', and 'state'
        # load CSV
        if file_path:
            self.load(file_path)
        # or initialise feature names
        else:
            if feature_names:
                self._observation_feature_names=feature_names
            else:
                sys.exit("An ObservationSequence requires at least one feature name.")
        
    def load ( self, file_path, feature_names=None):
        """
        Loads an observation sequence from a CSV file at @param file_path (str).
        @param feature_names (list): If a list of feature names (str) is provided, only those
            feature values will be loaded.
        """
        feature_names, observation_sequence = readObservationSequence(file_path, feature_names)
        self._observation_feature_names = feature_names
        for observation in observation_sequence:
            self.addObservation(observation)
    
    def save ( self, file_path, include_state=False ):
        """
        Saves the observation sequence as a CSV file at @param file_path (str).
        @param include_state (bool): Whether or not to include each observation's state information.
            This requires the _state attribute to be set for each Observation in self._observations.
        """
        serialiseObservationSequence( self.get(), file_path, self._observation_feature_names, include_state=include_state )
    
    def addObservation ( self, observation, position=None ):
        """
        Adds an observation to this sequence.
        @param observation: The Observation object to be added to this sequence.
        @param position: The target position for the Observation object in this sequence.
            Objects are appended to the end of the list by default.
        """
        assert isinstance(observation, Observation)
        if not len(observation.getValue()) == len(self._observation_feature_names):
            sys.exit("Cannot add Observation to ObservationSequence: Observation has %s feature values, while %s are expected." % (len(observation.getValue()), len(self._observation_feature_names)) )
        if position:
            self._observations.insert(position, observation)
        else:
            self._observations.append(observation)
            
    def get ( self ):
        """
        @return (list): The ordered list of Observation objects.
        """
        return self._observations
    
    def getList ( self, replaceNone=None ):
        """
        @param replaceNone (str/int/float): The value to replace empty feature values
            (None) with
        @return (list): The ordered list of observations, where each observation consists
            of a list of its feature values.
        """
        if replaceNone:
            return [ [e if e else replaceNone for e in o.getValue()] for o in self._observations ] 
        return [ observation.getValue() for observation in self._observations ]
    
    def getNumpyArray ( self, replaceNone=None ):
        """
        @param replaceNone (str/int/float): The value to replace empty feature values
            (None) with
        @return (np.array): The ordered list of observation as a Numpy array, where
            each observation consists of a Numpy array of its feature values.
        """
        if replaceNone:
            return np.array([ [e if e else replaceNone for e in o.getValue()] for o in self._observations ])
        return np.array([ np.array(observation.getValue()) for observation in self._observations ])
    
    def getObservation ( self, index ):
        return self._observations[index]
    
    def getFeatureNames ( self ):
        return self._observation_feature_names
        