#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Statistical Modelling of Human Trnaslation Processes
# Author: Samuel L�ubli (slaubli@inf.ed.ac.uk)

from __future__ import division

import sys, os, glob
import numpy as np
from collections import defaultdict
from sklearn.hmm import GaussianHMM, GMMHMM  # @UnresolvedImport
from sklearn.hmm import GMM # @UnresolvedImport

from adaptors.xml import *
from fileIO import readObservationSequences
from fileIO.fileIO import mkdir_p
from fileIO.xml import HMMSerialiser

# Ignore scikit-learn DeprecationWarnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

### HELPER FUNCTIONS
def getSubdirectories ( directory ):
    """
    @return (list): The name of all subdirectories in @param directory
    """
    return [subdirectory for subdirectory in os.listdir(directory) if os.path.isdir(os.path.join(directory,subdirectory))]

def trainModel ( training_sequences, num_states, num_comp, covariance_type ):
    """
    @return (GMMHMM): A GMM HMM model with @param num_states hidden states,
        @param num_comp GMM components per hidden state, and a covariance
        matrix of type @param covariance_type (full or diag), trained on
        @param training_sequences.
    """
    model = GMMHMM(num_states, n_mix=num_comp, covariance_type=covariance_type)
    return model.fit(training_sequences)

def saveModel ( model, path, feature_names=None, filename="model.xml" ):
    """
    Saves @param model as model.xml (or @param filename) at @param path
    Note: @param path: The path must be existing.
    """
    serialiser = HMMSerialiser(model, feature_names=feature_names)
    serialiser.saveXML(os.path.join(path, filename))

def writeLogprob ( logprob_file, total_likelihood_of_training_data, window_length, adaptor, num_states, num_comp, covariance_type ):
    """
    Writes the total likelihood of the training data @param
    total_likelihood_of_training_data to @param logprob_file.
    """
    logprob_file_entry = ",".join([window_length, adaptor, str(num_states), str(num_comp), covariance_type, str(total_likelihood_of_training_data)])
    logprob_file.write(logprob_file_entry + "\n")

def tagTrainingData ( model, training_sequences, observation_sequences):
    """
    Gets the most likely state sequence for each training_sequence through
    Viterbi decoding, and adds this information to the observation_sequence
    objects.
    @return (tuple): (total_likelihood_of_training_data, tagged_observation_sequences)
    """
    likelihood_of_training_data = 0.0 # log prob
    for i, training_sequence in enumerate(training_sequences):
        logprob, hidden_state_sequence = model.decode(training_sequence)
        likelihood_of_training_data += logprob
        for j, state in enumerate(hidden_state_sequence):
            observation = observation_sequences[i].getObservation(j)
            observation.setState( "H%s" % state )
    return likelihood_of_training_data, observation_sequences

def saveTaggedSequences ( tagged_observation_sequences, filenames, output_dir ):
    """
    Saves the tagged observation sequences in @param tagged_observation_sequences
    to a .csv file each, under a new folder named "tagged_training_data" in
    @param output_dir.
    """
    output_dir = os.path.join(base_path_out, 'tagged_training_data')
    mkdir_p(output_dir)
    for i, tagged_observation_sequence in enumerate(tagged_observation_sequences):
        filename = filenames[i].replace('.csv', '.tagged.csv')
        tagged_observation_sequence.save(os.path.join(output_dir, filename), include_state=True)

def writeRDataset ( stats_file, tagged_observation_sequences, filenames, window_length, adaptor, num_states, num_comp, covariance_type ):
    """
    Aggregates information on @param tagged_observation_sequences into an R dataset
    for further analysis. The dataset is appended to @param stats_file.
    """
    filename = []
    time = []
    all_state_occ = [] # number of observations tagged as state X
    all_state_occ_15min = [] # as above, in first 15 minutes the session
    all_state_phases = [] # number of coherent sequences (phases) per state, i.e., 11102200 => {0:2, 1:1, 2:1}
    all_state_phases_15min = [] # as above, in first 15 minutes of the session
    for i, tagged_observation_sequence in enumerate(tagged_observation_sequences):
        # counters
        state_occ = defaultdict(int) # number of observations tagged as state X
        state_occ_15min = defaultdict(int) # as above, in first 15 minutes the session
        state_phases = defaultdict(int) # number of coherent sequences (phases) per state, i.e., 11102200 => {0:2, 1:1, 2:1}
        state_phases_15min = defaultdict(int) # as above, in first 15 minutes of the session
        # aggregation
        observations = tagged_observation_sequence.get()
        #timestamp_start_plus_15min = observations[0].getStart() + 15*60*1000
        accumulated_time_in_ms = 0
        prev_state = None
        min15_in_ms = 15*60*1000
        min15_mark_passed = False
        for observation in observations:
            accumulated_time_in_ms += observation.getEnd() - observation.getStart()
            current_state = observation.getState()
            state_occ[current_state] += 1
            if current_state != prev_state and prev_state != None:
                state_phases[prev_state] += 1
            if accumulated_time_in_ms <= min15_in_ms: # 15 minutes in ms
                state_occ_15min[current_state] += 1
                if current_state != prev_state and prev_state != None:
                    state_phases_15min[prev_state] += 1
            elif not min15_mark_passed:
                state_phases_15min[current_state] += 1
                min15_mark_passed = True
            prev_state = current_state
        # fix for last observation in sequence
        state_phases[current_state] += 1
        if observation.getEnd() <= min15_in_ms: # 15 minutes in ms
            state_phases_15min[current_state] += 1
        # store values
        filename.append(filenames[i])
        time.append( accumulated_time_in_ms )
        all_state_occ.append(state_occ)
        all_state_occ_15min.append(state_occ_15min)
        all_state_phases.append(state_phases)
        all_state_phases_15min.append(state_phases_15min)
    # compose R code for dataframe
    # add filenames
    dataframe_name = "data." + ".".join([window_length, adaptor, "%sstates" % num_states, "%scomp" % num_comp, covariance_type])
    r_code = dataframe_name + " <- data.frame( filename=%s )" % getRVector(filename, to_string=True)
    r_code += " # filename of recorded translation session\n"
    # add session time in ms
    r_code += "%s$time <- %s # duration of translation session in ms\n" % (dataframe_name, getRVector(time))
    # number of observations tagged as state X for each state
    for state_name in range(0, num_states):
        state_name = 'H%s' % state_name
        r_code += "%s$%s <- %s " % (dataframe_name, "%s.occs" % state_name, getRVector(occ[state_name] for occ in all_state_occ))
        r_code += "# number of %s segments in the whole session\n" % state_name
    # number of observations tagged as state X for each state in the first 15 minutes
    for state_name in range(0, num_states):
        state_name = 'H%s' % state_name
        r_code += "%s$%s <- %s " % (dataframe_name, "%s.occs.15min" % state_name, getRVector(occ[state_name] for occ in all_state_occ_15min))
        r_code += "# number of %s segments in the first 15 minutes of the session\n" % state_name
    # number of coherent sequences (phases) per state, i.e., 11102200 => {0:2, 1:1, 2:1}
    for state_name in range(0, num_states):
        state_name = 'H%s' % state_name
        r_code += "%s$%s <- %s " % (dataframe_name, "%s.phases" % state_name, getRVector(occ[state_name] for occ in all_state_phases))
        r_code += "# number of coherent %s phases in the whole session\n" % state_name
    # number of coherent sequences (phases) per state (same as above) in the first 15 minutes
    for state_name in range(0, num_states):
        state_name = 'H%s' % state_name
        r_code += "%s$%s <- %s " % (dataframe_name, "%s.phases.15min" % state_name, getRVector(occ[state_name] for occ in all_state_phases_15min))
        r_code += "# number of coherent %s phases in the first 15 minutes of the session\n" % state_name
    # write R code to file
    stats_file.write(r_code + "\n")
    
def getRVector ( any_list, to_string=False ):
    """
    Returns an R-style vector given @param any_list. If @param to_string is True,
    all entries will be encapsulated in quotes (and thus be interpreted as strings
    in R).
    """
    if to_string:
        content = "','".join(any_list)
        return "c('%s')" % content
    else:
        content = ",".join([str(item) for item in any_list])
        return "c(%s)" % content

"""
Unsupervised training of GMM HMM models with the following parameters:
    - window length
    - adaptor type
    - number of states
    - number of GMM components per state

For each combination of the above parameters, this script stores
    - the model parameters (model.xml)
    - the tagged training sequences
    - the log likelihood of the training data (in the global file logprob.csv)
    - an R dataset for statistical analysis (in the global file data.R)

Usage: unsupervisedTrainingPipeline.py base_dir_input base_dir_output
input_dir has the following structure of subdirectories:
- 500ms
- 1000ms
- ...
    - Adaptor1
    - Adaptor2
    - ...
        - training_sequence1.csv
        - training_sequence2.csv
        - ...
"""

if len(sys.argv) == 3:
    base_dir_input = sys.argv[1]
    base_dir_output = sys.argv[2]
else:
    sys.exit("Usage: unsupervisedTrainingPipeline.py base_dir_input base_dir_output")

# CONSTANTS
MAX_NUM_STATES = 10 # max. number of HMM states
MAX_NUM_COMP = 10 # max. number of GMM mixture components
COVARIANCE_TYPE = 'full' # covariance type for GMMs (diag or full)

# start global file logprob.csv
logprob_file = open(os.path.join(base_dir_output, 'logprob.csv'), 'w')
logprob_file_header = "window_length,adaptor,states,components,covariance_type,logprob\n"
logprob_file.write(logprob_file_header)
# start global file data.R
stats_file = open(os.path.join(base_dir_output, 'data.R'), 'w')
stats_file_header  = "# Basic aggregations for segcats experiment series\n\n"
stats_file.write(stats_file_header)

# THE LOOP
# Iterate over window lengths
for window_length in getSubdirectories(base_dir_input):
    print "Window length: %s" % window_length
    # Iterate over adaptor types
    for adaptor in getSubdirectories(os.path.join(base_dir_input, window_length)):
        print "\tAdaptor: %s" % adaptor
        base_path_in = os.path.join(base_dir_input, window_length, adaptor)
        # read training observation sequences
        observation_sequences, filenames = readObservationSequences(os.path.join(base_path_in, '*.csv'), return_filenames=True)
        feature_names = observation_sequences[0].getFeatureNames()
        training_sequences = [ observation_sequence.getNumpyArray() for observation_sequence in observation_sequences ]
        # Iterate over number of hidden states
        for num_states in range(2, MAX_NUM_STATES+1):
            # Iterate over number of GMM mixture components
            for num_comp in range(1, MAX_NUM_COMP+1):
                print "\t\t Training model with %s hidden states and %s mixture components" % (num_states, num_comp)
                # create a new folder for this configuration
                base_path_out = os.path.join(base_dir_output, window_length, adaptor, "%s_states" % num_states, "%s_comp" % num_comp )
                mkdir_p(base_path_out)
                # train model
                model = trainModel(training_sequences, num_states, num_comp, COVARIANCE_TYPE)
                # save model
                saveModel(model, base_path_out, feature_names)
                # tag training data and get total likelihood
                total_likelihood_of_training_data, tagged_observation_sequences = tagTrainingData(model, training_sequences, observation_sequences)
                # save tagged training data
                saveTaggedSequences(tagged_observation_sequences, filenames, base_path_out)
                # write logprob to file
                writeLogprob(logprob_file, total_likelihood_of_training_data, window_length, adaptor, num_states, num_comp, COVARIANCE_TYPE)
                # write basic aggregations for further processing in R to file
                writeRDataset(stats_file, tagged_observation_sequences, filenames, window_length, adaptor, num_states, num_comp, COVARIANCE_TYPE)

logprob_file.close()
stats_file.close()