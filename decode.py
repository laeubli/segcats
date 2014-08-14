#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Statistical Modelling of Human Trnaslation Processes
# Author: Samuel Läubli (slaubli@inf.ed.ac.uk)

from __future__ import division

import sys, os, glob, argparse
import numpy as np
from collections import defaultdict
from sklearn.hmm import GaussianHMM, GMMHMM  # @UnresolvedImport
from sklearn.hmm import GMM # @UnresolvedImport

from adaptors.xml import *
from fileIO import readObservationSequence
from fileIO.fileIO import mkdir_p
from fileIO.xml import HMMReader

"""
Classifies each observation in an observation sequence (i.e., each window in a segmented translation session) as a HTP according to the provided HTP model.
Usage: decode.py -m model.xml input.csv output.csv
"""

# Ignore scikit-learn DeprecationWarnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

### HELPER FUNCTIONS
def loadModel ( path ):
    """
    @return model: A GMM HMM model according to @param path
    """
    return HMMReader(path).getModel()

def tagObservationSequence ( model, observation_sequence):
    """
    Gets the most likely state sequence for the provided @param observation_sequence
    through Viterbi decoding, as well as its log probability given @param model.
    @return (tuple): (log_likelihood, tagged_observation_sequence)
    """
    logprob, hidden_state_sequence = model.decode( observation_sequence.getNumpyArray() )
    for j, state in enumerate(hidden_state_sequence):
        observation = observation_sequence.getObservation(j)
        observation.setState( "H%s" % state )
    return logprob, observation_sequence

def getArgumentParser():
    """
    Return an ArgumentParser object to handle the arguments provided
    to this script.
    """
    parser = argparse.ArgumentParser(description="Classifies each observation in an observation sequence (i.e., each window in a segmented translation session) as a HTP according to the provided HTP model.\nUsage: decode.py -m model.xml input.csv output.csv\nSee --help for available options.")
    parser.add_argument("input", help="The the observation session, i.e., one CSV file.", type=str)
    parser.add_argument("output", help="The output file path.", type=str)
    parser.add_argument("-m", "--model", help="The HTP model trained with train.py (i.e., model.xml by default).", type=str)
    return parser

### MAIN ROUTINE ###

if __name__ == "__main__":
    # set up argument parser
    parser = getArgumentParser()
    args = vars( parser.parse_args() )
    # validate provided arguments
    observation_sequence_path = args['input']
    target_file = args['output']
    model_path = args['model']
    if not args['model']:
        sys.exit("No HTP model provided.\nRun train.py --help for further information.")
    # read model
    model = loadModel(model_path)
    # read observation sequence
    print "Tagging observation sequence..."
    observation_sequence = ObservationSequence(observation_sequence_path)
    # tag observation sequence
    log_likelihood, tagged_observation_sequence = tagObservationSequence(model, observation_sequence)
    print "Done. Log likelihood of the observation sequence given the provided model: %.8f." % log_likelihood
    # save tagged observation sequence to file
    tagged_observation_sequence.save(target_file, include_state=True)
    print "Tagged observation sequence written to %s." % target_file