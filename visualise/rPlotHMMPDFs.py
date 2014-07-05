#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Statistical Modelling of Human Translation Processes
# Author: Samuel Läubli (slaubli@inf.ed.ac.uk)

"""
Usage: python rPlotHMMPDFs.py model.xml > code.R

Reads a HMM model as produced by fileIO.xml and produces R code to plot
the probability densities for all features and states.

Takes a HMM model (XML file) as first argument, and prints the R code
to stdout.
"""

from __future__ import division

import sys
from lxml import etree # @UnresolvedImport

# helper functions
def removeNS (node_name):
    return node_name.split('}')[1]

# static R code (function definitions)
r_function_code_static = """# ***
# This code has been automatically generated using segcats.
# SCROLL TO THE VERY BOTTOM to see how toplot individual or multiple GMMs.
# ***

# Plotting functions
plotGMMs <- function(gmms, xmax=F, feature_names=F) {
  ### Combines multiple GMMs into a single plot
  ### @param gmmms: a list of gmm data frames as specified in plotGMM()
  
  # determine the number of plots per row (arrangement of plots inside plot)
  plots_per_row = ceiling(sqrt(length(gmms)))
  number_of_rows = ceiling(length(gmms)/plots_per_row)
  par(mfrow=c(number_of_rows,plots_per_row))
  
  # use the same maximum value on the x axis
  if (xmax==F) {
    xmax = 1 # minimum
    for (i in 1:length(gmms)) {
      # take highest mean plus twice its standard deviation as x max
      current_gmm = data.frame(gmms[i])
      m = max(order(current_gmm$mean))
      current_xmax = current_gmm$mean[m]
      current_xmax = current_xmax + 2 * sqrt(current_gmm$variance[m])
      if (current_xmax > xmax) {
        xmax = current_xmax
      }
    }
  }
  
  for (i in 1:length(gmms)) {
    feature_name = ifelse(feature_names == F, "", feature_names[i])[1]
    plotGMM(data.frame(gmms[i]), xmax=xmax, feature_name=feature_name)
  }
  
}
  
plotGMM <- function(gmm, xmax=F, feature_name="") {
  ### Plots a Gaussian mixture model stored in @param gmm
  ### @param gmm: a data.frame containing three columns: mean, variance, weight
  ### ---
  ### Example:
  ### gmm = data.frame( mean = c(3.13, 15.58, 39.06, 125.99)  )
  ### gmm$variance = c(168.75, 168.75, 168.75, 168.75)
  ### gmm$weight = c(0.25, 0.25, 0.25, 0.25)
  ### plotGMM(gmm)
  
  # define GMM probability density function (PDF)
  gmmf <- function(x) {
    y = 0
    for (i in 1:length(gmm$mean)) {
      y = y + gmm$weight[i] * dnorm(x, mean=gmm$mean[i], sd=sqrt(gmm$variance[i]))
    }
    return(y)
  }
  
  # set upper x limit for the plot (if not given)
  if (xmax==F) {
    # take highest mean plus twice its standard deviation as x max
    i = max(order(gmm$mean))
    xmax = gmm$mean[i]
    xmax = xmax + 2 * sqrt(gmm$variance[i])
  }
  plot(gmmf, xlim=c(0,xmax), ylab="Density", xlab="Observation", main=feature_name)
  
}
  
plotGMM <- function(gmm, xmax=F, feature_name="") {
  ### Plots a Gaussian mixture model stored in @param gmm
  ### @param gmm: a data.frame containing three columns: mean, variance, weight
  ### ---
  ### Example:
  ### gmm = data.frame( mean = c(3.13, 15.58, 39.06, 125.99)  )
  ### gmm$variance = c(168.75, 168.75, 168.75, 168.75)
  ### gmm$weight = c(0.25, 0.25, 0.25, 0.25)
  ### plotGMM(gmm)
  
  # define GMM probability density function (PDF)
  gmmf <- function(x) {
    y = 0
    for (i in 1:length(gmm$mean)) {
      y = y + gmm$weight[i] * dnorm(x, mean=gmm$mean[i], sd=sqrt(gmm$variance[i]))
    }
    return(y)
  }
  
  # set upper x limit for the plot (if not given)
  if (xmax==F) {
    # take highest mean plus twice its standard deviation as x max
    i = max(order(gmm$mean))
    xmax = gmm$mean[i]
    xmax = xmax + 2 * sqrt(gmm$variance[i])
  }
  plot(gmmf, xlim=c(0,xmax), ylab="Density", xlab="Observation", main=feature_name)
  
}
"""

if __name__ == "__main__":
    # exit if no file (HMM model) is provided
    if not sys.argv[1]:
        sys.exit("Usage: rPlotHMMPDFs.py model.xml")
    # read the model
    root_node = etree.parse(sys.argv[1]).getroot()
    # make sure that a GMM model was loaded
    assert root_node.get('type') == 'GMM'
    # print the static R code (function definitions)
    print r_function_code_static
    # iterate over states and print the GMMs for each feature
    state_names = []
    feature_names = []
    for state_node in root_node:
        state_name = state_node.get('name')
        if state_name != 'START':
            print
            print "# GMMs for State %s" % state_name
            if state_name not in state_names:
                state_names.append(state_name)
            for continuous_feature_node in state_node[1]:
                feature_name = continuous_feature_node.get('name')
                if feature_name not in feature_names:
                    feature_names.append(feature_name)
                means = []
                variances = []
                weights = []
                for gaussian_node in continuous_feature_node[0]:
                    means.append(float(gaussian_node.get('mean')))
                    variances.append(float(gaussian_node.get('variance')))
                    weights.append(float(gaussian_node.get('weight')))
                print "%s.%s = data.frame( mean = c(%s) )" % ( state_name, feature_name, ', '.join(['%.4f' % m for m in means]) )
                print "%s.%s$variance = c(%s)" % ( state_name, feature_name, ', '.join(['%.4f' % v for v in variances]) )
                print "%s.%s$weight = c(%s)" % ( state_name, feature_name, ', '.join(['%.4f' % w for w in weights]) )
                print
            print "GMMs.%s = list(%s)" % ( state_name, ', '.join(['%s.%s' % (state_name, feature_name) for feature_name in feature_names]) )
    # make the state and feature names available in R
    print
    print "# state and feature names"
    print 'state_names = c("%s")' % '", "'.join(state_names)
    print 'feature_names = c("%s")' % '", "'.join(feature_names)
    # print example code
    print
    print "# ***"
    print "# Example commands"
    print "# ***"
    print
    print "# Plot the GMM of feature %s in state %s" % (feature_name, state_name)
    print 'plotGMM(%s.%s, feature_name="%s")' % (state_name, feature_name, feature_name)
    print
    print "# Print all GMMs of state %s into a combined plot" % state_name
    print "plotGMMs(GMMs.%s, feature_names=feature_names)" % state_name
    print