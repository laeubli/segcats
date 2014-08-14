#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Statistical Modelling of Human Trnaslation Processes
# Author: Samuel Läubli (slaubli@inf.ed.ac.uk)

from __future__ import division

import sys, os, glob
import numpy as np
from sklearn.hmm import GaussianHMM, GMMHMM  # @UnresolvedImport

from adaptors.xml import *
from fileIO import readObservationSequences
from fileIO.fileIO import mkdir_p
from fileIO.xml import HMMSerialiser

#print "Extracting observations from XML..."

# Exctract observation sequences and save to CSV
source = "/Users/sam/Documents/ausbildung/uni/msc_ai/thesis/Data/TPR Raw/CFT13/Translog-II/P??_P??.xml"
#source = "/Users/sam/Documents/ausbildung/uni/msc_ai/thesis/Data/TPR Raw/CFT13/Translog-II/P01_P23.xml"
output_dir = "/Users/sam/Documents/ausbildung/uni/msc_ai/thesis/Models/MultivariateGaussianHMM/keyDown/A/P/"
adaptor = XMLAdaptorMultiWindow1()
for file_path in glob.glob(source):
    observation_sequence = adaptor.convert(file_path)
    if observation_sequence:
        observation_sequence.save(output_dir + "training_observations/" + os.path.basename(file_path) + '.csv')

print "Loading observations from CSV..."

# Load observation sequences from CSV
observation_sequences, filenames = readObservationSequences(output_dir + "training_observations/*.csv", return_filenames=True)
training_sequences = [ observation_sequence.getNumpyArray() for observation_sequence in observation_sequences ]

print "Training Multivariate Gaussian HMM model..."
n_components = 3
model = GaussianHMM(n_components, covariance_type="full", n_iter=10)
model.fit(training_sequences)

# save Gaussian HMM model to file
model_dir = output_dir + '%sstates/' % n_components
mkdir_p(model_dir)
serialiser = HMMSerialiser(model, feature_names=adaptor.getFeatures())
serialiser.saveXML(model_dir + 'model.xml')
 
print "Tagging observation sequences..."
tagged_sequences_dir = model_dir + "tagged_sequences/"
mkdir_p(tagged_sequences_dir)
for i, training_sequence in enumerate(training_sequences):
    hidden_state_sequence = model.predict(training_sequence)
    for j, state in enumerate(hidden_state_sequence):
        observation_sequences[i].getObservation(j).setState( "H%s" % state )
    # save tagged sequence to file
    filename = filenames[i].replace('.csv', '.tagged.csv')
    observation_sequences[i].save(tagged_sequences_dir + filename, include_state=True)

print "Training GMM-HMM model..."
n_components = 3
model = GMMHMM(n_components, n_mix=2, covariance_type="full", n_iter=100) # Multiple Gaussians (GMM) per State and Feature
model.fit(training_sequences)

# Save GMM HMM model to file
serialiser = HMMSerialiser(model, feature_names=adaptor.getFeatures())
serialiser.saveXML(model_dir + 'model_gmm.xml')