#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Statistical Modelling of Human Trnaslation Processes
# Author: Samuel L�ubli (slaubli@inf.ed.ac.uk)

from __future__ import division

from lxml import etree
import os, sys, glob, codecs, re, nltk

'''
Converts a TranslogII-style XML document into a CSV representation of event observations.
This adaptor only considers a single event type (EVENT_NAME) and exactly one of its
attributes (EVENT_ATTRIBUTE), which will be output as a single feature representing the
event.

Takes 1..* XML files as arguments and creates one observation file each (.obs) in the
current working directory.
'''

EVENT_NAME = 'fixation'
EVENT_ATTRIBUTE = 'duration'

def _fast_iter ( context, func, filestream ):
    '''
    Quickly iterates over all elements of an xml tree (lxml)
    See http://www.ibm.com/developerworks/xml/library/x-hiperfparse/
    '''
    for event, elem in context:
        func(elem, filestream)
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    del context

def process_log ( node, filestream ):
    global EVENT_ATTRIBUTE
    csv_line = []
    attribute = node.get(EVENT_ATTRIBUTE)
    # write csv line
    filestream.write('%s\n' % attribute)

# open all files provided as arguments
for filepath in sys.argv[1:]:
    # get file meta information
    filename = os.path.basename(filepath)
    # process xml file contents
    time_elapsed = []
    initialised = False
    with open(filepath, 'r') as xml_file:
        with open(filename + '.obs', 'w') as target_file:
            # write CSV header
            target_file.write('# %s attribute of %s events\n' % (EVENT_ATTRIBUTE, EVENT_NAME)) # feature definitions comment
            target_file.write('G\n') # feature definitions; use Weibull distribution for continuous features
            # process log events
            context = etree.iterparse(xml_file, events=('end',), tag=EVENT_NAME, recover=True)
            _fast_iter(context, process_log, target_file)