#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Statistical Modelling of Human Trnaslation Processes
# Author: Samuel Läubli (slaubli@inf.ed.ac.uk)

from __future__ import division

import sys, os, glob
import numpy as np
from sklearn.hmm import GaussianHMM, GMMHMM  # @UnresolvedImport
from sklearn.hmm import GMM # @UnresolvedImport

from adaptors.xml import *
from fileIO import readObservationSequences
from fileIO.fileIO import mkdir_p
from fileIO.xml import HMMSerialiser

# Ignore scikit-learn DeprecationWarnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

"""
Learns a GMM HMM with n hidden states and m mixture components from data (completely unsupervised).
In essence, the learning algorithm works as follows:
    Given:    - Training data (T) = 1..* Obseravtion Sequences (O) = 1..* Observations of 1..f features
    ---
    1.    KMeans clustering with K = n (number of states)
          Returns a <mean, variance> tuple for each dimension (one per feature f) for each state
    2.    EM training with the initialisations obtained from (1.)
          Returns a multivariate Gaussian HMM
    3.    Tag (T) with the model obtained from (2.)
    4.    Group all Observations in tagged (T) according to the state assigned in (3.)
          Returns one list of Observations per state
    5.    For each list of Observations (and thus each state), initialise a GMM with m mixture 
          components as follows:
            5.1.    KMeans clustering with K = m (number of mixture components)
            5.2.    Initialise GMM for current state with means and variances obtained from (5.1.)
            5.3.    Initialise GMM weights uniformly
    6.    Initialise a GMM HMM model as follows:
            6.1.    Transition probabilities = transition probabilities of previous HMM model
            6.2.    Observation probabilities = GMMs as obtained from (5.)
    7.    EM training with the initialisations obtained from (6.)
          Returns a GMM HMM model with m components
    8.    Iterate 4.--7. until the desired number of mixture components is reached and/or the
          increase of the likelihood of the training data doesn't increase significantly anymore.
"""

### CONSTANTS / SETTINGS

n_states = 3   # number of hidden states
m_components = 2   # number of mixture components for first GMM model
new_m = lambda m: m + 1 # function to increase m for subsequent GMM models
num_GMM_models = 9 # number of GMM models to train (= number of iterations of steps 4.--7.)

working_dir = "/Users/sam/Documents/ausbildung/uni/msc_ai/thesis/Models/GMMHMM/test/" # base working directory
training_data = working_dir + "training_observations/*.csv"

covariance_type = 'full' # covariance type for HMM models
num_EM_iterations = 5 # number of iterations for each EM training step (will be replaced by prob. increase threshold)

### HELPER FUNCTIONS

def saveModel ( hmm_model, folder_name, feature_names=None ):
    """Saves @param hmm_model as model.xml in a folder named @param folder name in the working_dir"""
    model_dir = working_dir + os.sep + folder_name
    mkdir_p(model_dir)
    serialiser = HMMSerialiser(hmm_model, feature_names=feature_names)
    serialiser.saveXML(model_dir + os.sep + 'model.xml')

def tagTrainingData ( hmm_model, training_sequences, observation_sequences, save=None, filenames=None ):
    """Tags all @param training_sequences according to @param hmm_model. Returns the total likelihood
    of the training data given the model, as well as a list of all observations in the training data
    grouped by state. If save is not None, the tagged sequences are saved in the folder @param save."""
    if save:
            save = working_dir + save
            mkdir_p(save)
    observations_per_state = [ list() for i in range(n_states) ]
    likelihood_of_training_data = 0.0 # log prob
    for i, training_sequence in enumerate(training_sequences):
        logprob, hidden_state_sequence = hmm_model.decode(training_sequence)
        likelihood_of_training_data += logprob
        for j, state in enumerate(hidden_state_sequence):
            observation = observation_sequences[i].getObservation(j)
            observation.setState( "H%s" % state )
            observations_per_state[state].append( observation.getValue() )
        # save tagged sequence to file
        if save:
            filename = filenames[i].replace('.csv', '.tagged.csv')
            observation_sequences[i].save(save + os.sep + filename, include_state=True)
    return likelihood_of_training_data, observations_per_state

### PROTOTYPE ROUTINE

print "Loading training data..."
# Load observation sequences from CSV
observation_sequences, filenames = readObservationSequences(training_data, return_filenames=True)
training_sequences = [ observation_sequence.getNumpyArray() for observation_sequence in observation_sequences ]

print "Training multivariate Gaussian HMM (base model)..."
# Implements (1.), (2.)
base_model = GaussianHMM(n_states, covariance_type=covariance_type, n_iter=num_EM_iterations)
base_model.fit(training_sequences)
# save base model
print "\tSaving base model to file..."
saveModel(base_model, 'base_model', observation_sequences[0].getFeatureNames())
# tag training data using base model
print "\tTagging training data using base model..."
# Implements (3.), (4.)
likelihood_of_training_data, observations_per_state = tagTrainingData( base_model, 
                                                                       training_sequences, 
                                                                       list(observation_sequences), # pass a copy 
                                                                       save='base_model/tagged_training_data', 
                                                                       filenames=filenames )
print "\tTotal log lokelihood of the training data according to base model: %.4f" % likelihood_of_training_data

previous_model = base_model
for i in range(num_GMM_models):
    print "Training GMM HMM #%s (%s components)" % (i+1, m_components)
    # Fit GMM to observations currently assigned to each state
    print "\tInitialising GMM..."
    # Implements (5.)
    GMMs = []
    for state_index in range(n_states):
        g = GMM( n_components=m_components,
                   covariance_type=covariance_type,
                   n_iter=0, # this initialises the GMMs without optimising them through EM; this is done later in the GMM HMM model
                  ) # n_iter = 5(...) better??
        g.fit(observations_per_state[state_index])
        GMMs.append(g)
    # Initialise and train new GMM HMM model
    print "\tTraining GMM HMM model..."
    # Implements (6.), (7.)
    gmm_model = GMMHMM( n_components=n_states,
                        n_mix=m_components,
                        startprob=previous_model.startprob_,
                        transmat=previous_model.transmat_,
                        gmms=GMMs,
                        covariance_type=covariance_type,
                        n_iter=num_EM_iterations,
                        init_params='' # initialisation through previous model and GMMs!
                       )
    # save base model
    print "\tSaving model to file..."
    saveModel(gmm_model, 'gmm_model_%sstates' % m_components, observation_sequences[0].getFeatureNames())
    # tag training data using new model
    print "\tTagging training data using base model..."
    likelihood_of_training_data, observations_per_state = tagTrainingData( gmm_model, 
                                                                           training_sequences, 
                                                                           list(observation_sequences), # pass a copy 
                                                                           save='gmm_model_%sstates/tagged_training_data' % m_components, 
                                                                           filenames=filenames )
    print "\tTotal log lokelihood of the training data according to current model: %.4f" % likelihood_of_training_data
    # update meta parameters for next iteration
    m_components = new_m(m_components)
    previous_model = gmm_model
    
# model.fit(training_sequences)
# 
# # save Gaussian HMM model to file
# model_dir = output_dir + '%sstates/' % n_components
# mkdir_p(model_dir)
# serialiser = HMMSerialiser(model, feature_names=adaptor.getFeatures())
# serialiser.saveXML(model_dir + 'model.xml')
#  
# print "Tagging observation sequences..."
# tagged_sequences_dir = model_dir + "tagged_sequences/"
# mkdir_p(tagged_sequences_dir)
# for i, training_sequence in enumerate(training_sequences):
#     hidden_state_sequence = model.predict(training_sequence)
#     for j, state in enumerate(hidden_state_sequence):
#         observation_sequences[i].getObservation(j).setState( "H%s" % state )
#     # save tagged sequence to file
#     filename = filenames[i].replace('.csv', '.tagged.csv')
#     observation_sequences[i].save(tagged_sequences_dir + filename, include_state=True)
# 
# print "Training GMM-HMM model..."
# n_components = 3
# model = GMMHMM(n_components, n_mix=2, covariance_type="full", n_iter=100) # Multiple Gaussians (GMM) per State and Feature
# model.fit(training_sequences)
# 
# # Save GMM HMM model to file
# serialiser = HMMSerialiser(model, feature_names=adaptor.getFeatures())
# serialiser.saveXML(model_dir + 'model_gmm.xml')