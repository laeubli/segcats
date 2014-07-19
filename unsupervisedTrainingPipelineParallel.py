#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Statistical Modelling of Human Trnaslation Processes
# Author: Samuel L�ubli (slaubli@inf.ed.ac.uk)

from __future__ import division

import sys, os, glob, subprocess
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
BASE_SCRIPT_FULL_PAHT = os.path.join( os.path.dirname(os.path.realpath(__file__)), "unsupervisedTraining.py")
MAX_NUM_STATES = 10 # max. number of HMM states
MAX_NUM_COMP = 10 # max. number of GMM mixture components
COVARIANCE_TYPE = 'full' # covariance type for GMMs (diag or full)

# start global file logprob.csv
logprob_file_path = os.path.join(base_dir_output, 'logprob.csv')
with open(logprob_file_path, 'w') as logprob_file:
    logprob_file_header = "window_length,adaptor,states,components,covariance_type,logprob\n"
    logprob_file.write(logprob_file_header)
# start global file data.R
stats_file_path = os.path.join(base_dir_output, 'data.R')
with open(stats_file_path, 'w') as stats_file:
    stats_file_header  = "# Basic aggregations for segcats experiment series\n\n"
    stats_file.write(stats_file_header)

# THE LOOP
# Iterate over window lengths
for window_length in getSubdirectories(base_dir_input):
    # Iterate over adaptor types
    for adaptor in getSubdirectories(os.path.join(base_dir_input, window_length)):
        base_path_in = os.path.join(base_dir_input, window_length, adaptor)
        path_training_data = os.path.join(base_path_in, '*.csv')
        #path_training_data = "'" + path_training_data + "'"
        # Iterate over number of hidden states
        for num_states in range(2, MAX_NUM_STATES+1):
            # train model with num_states and 1..MAX_NUM_COMP mixture components in parallel
            subprocesses = []
            for num_comp in range(1, MAX_NUM_COMP+1):
                path_out = os.path.join(base_dir_output, window_length, adaptor, "%s_states" % num_states, "%s_comp" % num_comp )
                arguments = [path_training_data, path_out, logprob_file_path, stats_file_path, window_length, adaptor, str(num_states), str(num_comp), COVARIANCE_TYPE]
                subprocesses.append( subprocess.Popen(['python', BASE_SCRIPT_FULL_PAHT] + arguments) )
            exit_codes = [p.wait() for p in subprocesses]
            