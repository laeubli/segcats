#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import matplotlib as mpl
mpl.use('PDF') # ensures that plotting PDF densities works in environments where DISPLAY is not set

import sys, os, errno
from fileIO import readObservationSequences, serialiseObservationSequence
from shared import mean, variance
from model import SingleGaussianHMM
from mimetypes import init

'''
Trains models with 2, 3, 4, and 5 hidden states on n observation sequences.
Note: Must be run from the main folder of the experiment, where all training
observation sequences are stored as CSV files in a folder called training_observations.
'''

def mkdir_p(path):
    '''
    Creates a directory and overwrites existing directories with the same name (mkdir -p).
    '''
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

# read training observation sequences
training_sequences = []
observation_sequences, filenames = readObservationSequences('training_observations/*.csv', return_filenames=True)
original_filenames = []
# transform and collect all observations in single list (for initial mean/variance estimation)
all_observation_values = []
for observation_sequence in observation_sequences:
    training_observation_sequence = []
    for observation in observation_sequence:
        value = observation.getValue()
        all_observation_values.append(value[0])
        training_observation_sequence.append(value)
    training_sequences.append(training_observation_sequence)
# estimate global variance
global_variance = variance(all_observation_values)

# MAIN LOOP: for 2, 3, 4, and 5 hidden states
number_of_hidden_states = [2,3,4,5]
for i in number_of_hidden_states:
    sys.stdout.write('TRAINING %s-STATE MODEL\n***\n' % i)
    sys.stderr.write('TRAINING %s-STATE MODEL\n***\n' % i)
    # define state names
    states = [ 'H'+str(j+1) for j in range(0,i) ]
    states = ['START'] + states + ['END']
    # define initial means for each state's Gaussian
    initial_obs_probs = []
    min_ = min(all_observation_values)
    max_ = max(all_observation_values)
    range_ = max_ - min_
    part_width = range_ / (i-1)
    for j in range(0,i):
        mean = min_ + (j*part_width) # was: min_ + ( (j*part_width) + (part_width/2) )
        variance = global_variance
        initial_obs_probs.append( (mean, variance) )
    # add start and end observation probabilities (None)
    initial_obs_probs = [(None, None)] + initial_obs_probs + [(None, None)]
    # train HMM
    model = SingleGaussianHMM(
                    states=states, 
                    observation_sequences=training_sequences, 
                    initial_observation_probabilities=initial_obs_probs, 
                    topology='fully-connected', 
                    training_iterations=7, 
                    verbose=True
                    )
    # create a folder to store all results
    results_dir = '%sstates/' % i
    mkdir_p(results_dir)
    # store model
    model.save(results_dir + 'model.xml')
    model.visualisePDFHistory(results_dir + 'probability_densities.pdf', max_)
    # tag all observation sequences
    tagged_sequences_dir = results_dir + 'tagged_sequences/'
    mkdir_p(tagged_sequences_dir)
    for i, obs_seq in enumerate(training_sequences):
        most_probable_state_seq = model.viterbiProbability(obs_seq)[0]
        for j, state in enumerate(most_probable_state_seq[1:-1]): # exclude START and END states
            observation_sequences[i][j].setState( states[state] )
        # save tagged sequence to file
        filename = filenames[i].replace('.csv', '.tagged.csv')
        serialiseObservationSequence(observation_sequences[i], tagged_sequences_dir + filename, ['value'], include_state=True)