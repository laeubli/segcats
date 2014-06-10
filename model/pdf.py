#!/usr/bin/env python
"""
Defines the HMM feature functions (PDFs).
IMPORTANT: All probabilities are natural logarithms, normally derived through log() from the shared packet
"""

from __future__ import division

import sys, math
import numpy as np
from shared import *
from scipy import stats
from matplotlib import pyplot as plt

class AbstractFeature(object):
    """
    A continuous feature holds the parameters of a probability density function (e.g., mean 
    and variance for a Gaussian PDF) for each state.
    """
    
    def __init__ ( self, states, observations, plotInitialObservationProbs=False ):
        """
        @param states (list): the HMM states
        @param observations (list): all feature values (global)
        @param plotInitialObservationProbs (bool): If true, the initial observation probabilities
            derived from the training observations will be plotted, alongside with the respective
            PDF (if applicable).
        """
        self._states = states
        self._observation_probabilities = []
        self._plotInitialObservationProbs = plotInitialObservationProbs
    
    def getStateIndex ( self, state ):
        try:
            return self._states.index(state)
        except:
            sys.exit('Error: State "%s" is undefined in this model' % state)
    
    def getStates ( self ):
        return self._states
    
    def getProbability ( self, state, observation ):
        """
        Returns the probability P(observation|state).
        """
        pass #@override in subclass
    
    def getStateProbabilities ( self, observation ):
        """
        Returns the probability P(observation|state) for all states in self._states.
        """
        pass #@override in subclass
    
    def plot ( self, state ):
        """
        Plots the observation probabilitie(s) for a state.
        """
        pass #@override in subclass



class AbstractContinuousFeature(AbstractFeature):
    """
    A continuous feature holds the parameters of a probability density function (e.g., mean 
    and variance for a Gaussian PDF) for each state.
    """
    
    def __init__ ( self, states, observations, plotInitialObservationProbs=False ):
        AbstractFeature.__init__(self, states, observations, plotInitialObservationProbs)
        ###
        # Estimate initial observation probabilities
        initial_parameters = self._estimatePDFParameters(observations)
        # Use the same initial estimate for all states
        for state in self._states:
            if state not in ['START', 'END']: # START and END states are non-emiting
                self._observation_probabilities.append(initial_parameters)
            else:
                self._observation_probabilities.append(None)
        # Plot initial PDF
        if plotInitialObservationProbs:
            self.plot(1, observations) # We can plot any state besides START and END since all have 
                                       # the same observation probabilities at this point.
    
    def getProbability ( self, state, observation ):
        """
        Returns the probability P(observation|state).
        @param state (str/int): The name (str) or index (int) of the state. Providing
            the index rather than the name of the state speeds up the lookup
        @param observation (float): The observed value
        @return (float): the probability of seeing @param observation in @param state
        """
        if state in ['START','END', 0, len(self._states)-1]:
            return None
        if isinstance(state, str):
            state = self.getStateIndex(state)
        assert isinstance(observation, float)
        return self._applyPDF(self._observation_probabilities[state], observation)

    def getStateProbabilities ( self, observation ):
        """
        Returns the probability P(observation|state) for all states in self._states.
        @param observation (float): The observed value
        @return list: A list of state-observation probabilities; the order corresponds
            to self.getStates()
        """
        assert isinstance(observation, float)
        return [self.getProbability(state, observation) for state in range(len(self._states))]
    
    def plot ( self, state, observations=None ):
        """
        Plots the PDF for a state. If observations are provided, they are interleaved with
        the PDF as a histogram.
        @param state (str/int): The name (str) or index (int) of the state; providing 
            an index speeds up the lookup.
        @param observations (list): The observations (floats) that the PDF parameters were
            estimated from.
        """
        pass #@override in subclass
        
    
    def _plot ( self, pdf, state, observations=None ):
        """
        Helper for self._plot()
        @param pdf (function): The scipy PDF function, e.g., scipy.stats.norm.pdf
        """
        if isinstance(state, str):
            state = self.getStateIndex(state)
        # histogram of observations
        plt.hist(observations, normed=True, bins=25, color='#D0E0EB')
        # PDF
        side_padding = max(observations)*0.05 # leave some space at the sides
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin-side_padding,xmax+side_padding,100)
        p = pdf(x, *self._observation_probabilities[state])
        plt.plot(x, p, '#A2005B', linewidth=2)
        # format the plot
        ax = plt.axes()
        ax.set_title("Probability Density")
        ax.set_xlabel("Observed Value")
        ax.set_ylabel("Probability")
        plt.xlim([min(observations)-side_padding, max(observations)+side_padding])
        #plt.ylim(0, max(observations)/len(observations)*1.05)
        plt.show()
    
    def _estimatePDFParameters( self, observations ):
        """
        Estimates the PDF parameters (e.g., mean and variance for Gaussian) from a list of observed values.
        @param observations (list): the observed values (floats) to estimate the PDF parameters from
        @return (list): The PDF parameters (floats) packed into a list
        """
        pass #@override in subclass

    def _applyPDF ( self, parameters, observation):
        """
        Returns the probability of @param observation (float) by applying the PDF with the provided parameters.
        @param parameters (list of floats): the parameters of the PDF, e.g., mean and variance for a Gaussian PDF
        @param observation (float): the observed value
        @return float: the probability of @param observation given @param parameters according to this PDF
        """
        pass #@override in subclass
        


class DiscreteFeature(AbstractFeature):
    """
    A discrete feature holds an observation vocabulary of valid observations and an observation
    probability for each observation-state pair.
    """
    
    def __init__ ( self, states, observations, plotInitialObservationProbs=False ):
        """
        @param states (list): the HMM states
        @param observations (dict): all feature values (i.e., the vocabulary) and their global counts
        """
        AbstractFeature.__init__(self, states, observations, plotInitialObservationProbs)
        ###
        self._vocabulary = observations.keys() # discrete features have an observation vocabulary
        for state in self._states:
            self._observation_probabilities.append([])
        # MLE estimate for initial observation probabilities
        total_observations = sum(observations.values())
        for observation in self._vocabulary:
            initial_observation_probability = observations[observation] / total_observations
            initial_observation_probability = log(initial_observation_probability) # logarithmise
            for i, state in enumerate(self._states):
                if state not in ['START', 'END']: # START and END states are non-emiting
                    self._observation_probabilities[i].append(initial_observation_probability)
                else:
                    self._observation_probabilities[i].append(None)
        # Plot the initial observation probabilities (if applicable)
        if self._plotInitialObservationProbs:
            self.plot(1) # We can plot any state besides START and END since all have the same
                         # observation probabilities at this point.
    
    def getObservationIndex ( self, observation ):
        try:
            return self._vocabulary.index(observation)
        except:
            sys.exit('Error: Observation "%s" was never seen in training.' % observation)
    
    def getVocabulary ( self ):
        return self._vocabulary
    
    def getProbability ( self, state, observation ):
        """
        Returns the probability P(observation|state).
        Both @param state and @param observation can either be indizes (int) of this object's
        observation probability matrix (self._observation_probabilities) or the name (str) of
        the state and/or observation. Providing indizes speeds up the lookup.
        @return (float): the probability of seeing @param observation in @param state
        """
        if isinstance(state, str):
            state = self.getStateIndex(state)
        if isinstance(observation, str):
            observation = self.getObservationIndex(observation)
        return self._observation_probabilities[state][observation]
    
    def getStateProbabilities ( self, observation ):
        """
        Returns the probability P(observation|state) for all states in self._states.
        @param observation (str/int): The name or index of an observation. Providing
            an index speeds up the lookup.
        @return list: A list of state-observation probabilities; the order corresponds
            to self.getStates()
        """
        if isinstance(observation, str):
            observation = self.getObservationIndex(observation)
        return [self._observation_probabilities[state][observation] for state in range(len(self._states))]
    
    def plot ( self, state ):
        """
        Plots the probabilities for each observation given a state.
        @param state (str/int): The name (str) or index (int) of the state; providing 
            an index speeds up the lookup.
        """
        if isinstance(state, str):
            state = self.getStateIndex(state)
        # get labels and probabilities
        labels = self._vocabulary
        probs = [math.exp(prob) for prob in self._observation_probabilities[state]] # convert log probs back to normal probs for the plot
        # format the plot
        pos = np.arange(len(labels))
        width = 0.75
        ax = plt.axes()
        ax.set_xticks(pos + (width/2))
        ax.set_xticklabels(labels)
        ax.set_title("Observation Probabilities")
        ax.set_xlabel("Observation")
        ax.set_ylabel("Probability")
        plt.bar(pos, probs, width, color=['#88ABC2','#D0E0EB'])
        plt.xlim([-0.5, len(labels)+0.5])
        plt.ylim(0, max(probs)+max(probs)*0.05)
        plt.show()

class GaussianFeature(AbstractContinuousFeature):
    """
    A Gaussian feature holds the mean and variance of a Gaussian for each state.
    """
    
    def __init__ (self, states, observations, plotInitialObservationProbs=False ):
        """
        @param states (list): the HMM states
        @param observations (dict): all feature values (i.e., the vocabulary) and their global counts
        """
        AbstractContinuousFeature.__init__(self, states, observations, plotInitialObservationProbs)
    
    def _estimatePDFParameters( self, observations ):
        """
        Estimates the mean and variance of a single Gaussian from a list of observed values.
        @param observations (list): the observed values (floats) to estimate the mean and std from
        @return (list): The mean and variance (floats) packed into a list
        """
        mean, std = stats.norm.fit(observations)
        return [mean, std]

    def _applyPDF ( self, parameters, observation):
        """
        Returns the probability of @param observation (float) by applying the PDF with the provided
            parameters.
        @param parameters (list of floats): the parameters of the PDF, i.e., mean and std
        @param observation (float): the observed value
        @return float: the probability of @param observation given @param parameters according to the
            Gaussian distribution
        """
        assert isinstance(observation, float)
        mean = parameters[0]
        std = parameters[1]
        return stats.norm.logpdf(observation, mean, std)
    
    def plot ( self, state, observations=None ):
        """
        Plots the PDF for a state. If observations are provided, they are interleaved with
        the PDF as a histogram.
        @param state (str/int): The name (str) or index (int) of the state; providing 
            an index speeds up the lookup.
        @param observations (list): The observations (floats) that the PDF parameters were
            estimated from.
        """
        self._plot(stats.norm.pdf, state, observations)


class WeibullFeature(AbstractContinuousFeature):
    """
    A Weibull feature holds the shape (k) and scale (l) parameters of a Weibull distribution
    for each state.
    """
    
    def __init__ (self, states, observations, plotInitialObservationProbs=False ):
        """
        @param states (list): the HMM states
        @param observations (dict): all feature values (i.e., the vocabulary) and their global counts
        """
        AbstractContinuousFeature.__init__(self, states, observations, plotInitialObservationProbs)
    
    def _estimatePDFParameters( self, observations ):
        """
        Estimates the shape (k) and scale (l) parameters of a Weibull distribution from a list of 
        observed values.
        @param observations (list): the observed values (floats) to estimate the shape and scale 
            parameters from
        @return (list): The shape and scale parameters (floats) packed into a list
        """
        quantiles = np.percentile(observations, [10, 30, 50, 70, 90]) # the weibull_min.fit function expects quantiles rather than the actual observations
        weibull_params = stats.weibull_min.fit(quantiles, floc=0) # floc=0 keeps the location fixed at zero,
            # i.e, we assume that all observations are positive (lower bound = 0).
        return weibull_params

    def _applyPDF ( self, parameters, observation):
        """
        Returns the probability of @param observation (float) by applying the PDF with the provided
            parameters.
        @param parameters (list of floats): the parameters of the PDF, i.e., shape (k) and scale (l)
        @param observation (float): the observed value
        @return float: the probability of @param observation given @param parameters according to the
            Weibull
        """
        assert isinstance(observation, float)
        return stats.weibull_min.logpdf(observation, *parameters)
    
    def plot ( self, state, observations=None ):
        """
        Plots the PDF for a state. If observations are provided, they are interleaved with
        the PDF as a histogram.
        @param state (str/int): The name (str) or index (int) of the state; providing 
            an index speeds up the lookup.
        @param observations (list): The observations (floats) that the PDF parameters were
            estimated from.
        """
        self._plot(stats.weibull_min.pdf, state, observations)

