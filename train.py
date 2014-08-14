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
from fileIO.xml import HMMSerialiser

"""
Unsupervised training of a GMM-HMM HTP model.
Usage: train.py training_observations/*.csv
"""

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

def saveModel ( model, path, feature_names=None ):
    """
    Saves @param model as model.xml (or @param filename) at @param path
    Note: @param path: The path must be existing.
    """
    serialiser = HMMSerialiser(model, feature_names=feature_names)
    serialiser.saveXML(path)

def getArgumentParser():
    """
    Return an ArgumentParser object to handle the arguments provided
    to this script.
    """
    parser = argparse.ArgumentParser(description="Trains a statistical model of human translation processes from 1..* training observation sequences, as obtained through extract.py. See --help for available options.")
    parser.add_argument("source", help="The training observation(s), i.e., 1..* CSV files.", type=str, nargs='*')
    parser.add_argument("-o", "--destination", help="The target path where the trained model will be written to. The default is 'model.xml' in the current working directory.", default='model.xml', type=str)
    parser.add_argument("-k", "--states", default=10, help="The number of HMM states (HTPs). Default: 10.", type=int)
    parser.add_argument("-m", "--components", default=10, help="The number of Gaussian mixture components per state (HTP). Default: 10.", type=int)
    return parser

### MAIN ROUTINE ###

if __name__ == "__main__":
    # set up argument parser
    parser = getArgumentParser()
    args = vars( parser.parse_args() )
    # make sure that at least one training observation sequence is provided
    source = args['source']
    if source == []:
        sys.exit("Need at least one training observation sequence to train a model.\nRun train.py --help for further information.")
    # check output directory
    output_path = args['destination']
    # read training observation sequences
    observation_sequences = []
    filenames = []
    for filepath in source:
        observation_sequences.append( ObservationSequence(filepath) )
        filenames.append( os.path.basename(filepath) )
    # read feature names
    feature_names = observation_sequences[0].getFeatureNames()
    # format observation features
    training_sequences = [ observation_sequence.getNumpyArray() for observation_sequence in observation_sequences ]
    # train model
    print "Training GMM-HMM model with %s states (HTPs) and %s components per state on %s observation sequences..." % (args['states'], args['components'], len(observation_sequences))
    model = trainModel(training_sequences, args['states'], args['components'], 'full')
    print "Done. Model written to %s" % output_path
    # save model
    saveModel(model, output_path, feature_names)