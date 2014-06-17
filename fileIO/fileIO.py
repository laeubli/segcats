#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Provides file input/output functionality.
"""
import sys, os, glob, csv
from adaptors.observation import *

def serialiseObservationSequence ( observation_sequence, file_path, feature_names, include_state=False ):
    """
    Encodes an observation sequence as a CSV file and writes it to disk.
    @param observation_sequence (list): the list of Observation objects to be encoded
    @param file_path (str): the target file path to store the CSV file at
    @param feature_names (list): a list of strings to describe each feature value of
        the observations. The length of each Observation object'ss value must be equal 
        to the number of names provided. Names must not include 'start', 'end', and/or
        any duplicates.
    @param include_state (bool): If true, the observation's state (ass assigned by a
        HMM model) will be included.
    """
    # check feature_names:
    if 'start' in feature_names or 'end' in feature_names:
        sys.exit("Error saving observation sequence: 'start' and 'end' must not be used as feature names.")
    if len(feature_names) != len(set(feature_names)):
        sys.exit("Error saving observation sequence: no feature name can be used twice.")
    # encode and save observation sequence
    with open(file_path, 'wb') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"')
        # write header
        header = ['start', 'end'] + feature_names
        if include_state:
            header += ['state']
        writer.writerow(header)
        # write observations
        for observation in observation_sequence:
            assert isinstance(observation, Observation)
            start = observation.getStart()
            end = observation.getEnd()
            features = observation.getValue()
            row = [start, end] + features
            if include_state:
                row += [observation.getState()]
            writer.writerow(row)
        
def readObservationSequences ( path, features=None, return_filenames=False ):
    """
    Reads 1..* CSV-encoded observation sequences and returns them as a list of 
    observation sequences.
    @param file_path (str): The path to the observation file(s). Accepts wildcards such
        as /foo/bar/*.obs.
    @param features (list): The names of all features (CSV row headers) to be returned.
        Returns all features (i.e., all rows except for 'start' and 'end') by default.
    @param return_filenames (bool): Whether or not to also return a list of the file
        names of all read-in original files.
    @return (list): A list of observation sequence, where each observation sequence
        consists of a list of Observation objects; plus the list of file names if
        @param return_filenames is True.
    """
    observation_sequences = []
    file_names = []
    for file_path in glob.glob(path):
        observation_sequences.append( readObservationSequence(file_path, features) )
        file_names.append( os.path.basename(file_path) )
    if return_filenames:
        return observation_sequences, file_names
    else:
        return observation_sequences

def readObservationSequence ( file_path, features=None ):
    """
    Reads a single CSV-encoded observation sequence.
    @param file_path (str): The path to the CSV file
    @param features (list): The names of all features (CSV row headers) to be returned.
        Returns all features (i.e., all rows except for 'start' and 'end') by default.
    @return (list): A list of observations of type Observation
    """
    observation_sequence = []
    with open(file_path, 'rb') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        # read header
        field_names = reader.next()
        relevant_row_indices = [] # the index and order of the features to be extracted
        assert 'start' in field_names
        start_index = field_names.index('start')
        assert 'end' in field_names
        end_index = field_names.index('end')
        try:
            if features == None:
                for field_name in field_names:
                    if field_name not in ['start', 'end']:
                        relevant_row_indices.append(field_names.index(field_name))
            else:
                for feature in features:
                    relevant_row_indices.append(field_names.index(feature))
        except ValueError:
            sys.exit('Error reading observation sequence: Feature "%s" not found in file %s' % (feature, file_path))
        # read observations
        for row in reader:
            start = int( row[start_index] )
            end = int( row[end_index] )
            value = []
            for i in relevant_row_indices:
                try: value.append( int(row[i]) )
                except ValueError:
                    try: value.append( float(row[i]) )
                    except ValueError:
                        value.append( row[i] ) # as str
            observation_sequence.append( Observation(start, end, value) )
    return observation_sequence